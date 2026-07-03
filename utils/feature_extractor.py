"""Spectral feature extraction for the Akylai Keyword-Spotting project.

This module turns raw audio clips into a fixed-length table of classical
spectral features (MFCC + dynamics, mel energies, spectral-shape descriptors,
chroma, ZCR, RMS) suitable for *any* tabular classifier.

It is built around two reusable classes:

* :class:`SpectralFeatureExtractor` -- the pure DSP front-end. Given a single
  waveform it returns one fixed-length feature vector. It is deliberately free
  of any I/O so it can be shared between *offline* dataset processing and
  *online* streaming inference (see ``utils.stream_preprocessor``).

* :class:`DatasetFeatureExtractor` -- the batch orchestrator. It reads a source
  dataset (from the Hugging Face Hub or local disk), runs the front-end over
  every clip in parallel, carries selected metadata columns through, and writes
  the resulting feature table either to local parquet or to the Hub.

Everything is driven by a YAML config (see ``config.yaml``); run it with::

    python -m utils.feature_extractor --config config.yaml

Keeping the front-end configuration identical between training and inference is
critical: any mismatch (sample rate, FFT size, mel count, ...) silently
destroys accuracy ("training/serving skew").
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from joblib import Parallel, delayed


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class FrontendConfig:
    """Parameters of the DSP front-end.

    These values must be *identical* at training and inference time. They are
    intentionally explicit (no hidden library defaults) for reproducibility.

    Attributes:
        sample_rate: Target sample rate in Hz; audio is resampled to it.
        n_fft: STFT window length in samples (``512`` ~= 32 ms at 16 kHz).
        hop_length: STFT hop in samples (``160`` ~= 10 ms at 16 kHz).
        n_mfcc: Number of MFCC coefficients to keep.
        n_mels: Number of mel filter-bank channels.
        fmin: Lowest mel-filter frequency in Hz.
        fmax: Highest mel-filter frequency in Hz; ``None`` -> Nyquist.
        groups: Ordered list of feature groups to compute. Allowed values:
            ``mfcc, delta, delta2, mel, contrast, chroma, centroid,
            bandwidth, rolloff, flatness, zcr, rms``.
        aggregate: Time-aggregation statistics applied per channel.
    """

    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160
    n_mfcc: int = 20
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float | None = None
    groups: Sequence[str] = field(default_factory=lambda: [
        "mfcc", "delta", "delta2", "mel", "contrast", "chroma",
        "centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms",
    ])
    aggregate: Sequence[str] = field(default_factory=lambda: ["mean", "std"])


@dataclass
class DatasetConfig:
    """Where the *raw audio* comes from.

    Attributes:
        source: ``"hf"`` to load from the Hugging Face Hub, ``"local"`` to load
            a ``save_to_disk`` directory from local disk.
        repo_or_path: HF repo id (e.g. ``"aiacademy-kg/kws-raw"``) or local path.
        name: Optional HF dataset config name.
        split: Which split to read (e.g. ``"train"``).
        audio_col: Name of the column that holds the audio.
        keep_cols: Source columns to copy verbatim into the output table
            (e.g. ``label``, ``source``). They are never treated as features.
    """

    source: str = "hf"
    repo_or_path: str = "aiacademy-kg/kws-raw"
    name: str | None = None
    split: str = "train"
    audio_col: str = "audio"
    keep_cols: Sequence[str] = field(default_factory=lambda: ["label", "source"])


@dataclass
class OutputConfig:
    """Where the resulting feature table is written.

    Attributes:
        sink: ``"local"`` to write a parquet file, ``"hf"`` to push a dataset
            to the Hub.
        path: Local parquet path (used when ``sink == "local"``).
        hf_repo: Target Hub repo id (used when ``sink == "hf"``).
        private: Whether the pushed Hub dataset is private.
    """

    sink: str = "local"
    path: str = "features.parquet"
    hf_repo: str = "aiacademy-kg/kws-dataset"
    private: bool = False


@dataclass
class ProcessingConfig:
    """Parallelism settings for batch extraction.

    Attributes:
        n_jobs: Number of parallel worker processes.
        chunk_size: Number of clips handed to a worker at a time.
    """

    n_jobs: int = 6
    chunk_size: int = 1_000


@dataclass
class ExtractionConfig:
    """Top-level config bundling all sections."""

    dataset: DatasetConfig
    frontend: FrontendConfig
    output: OutputConfig
    processing: ProcessingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExtractionConfig":
        """Build an :class:`ExtractionConfig` from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            frontend=FrontendConfig(**raw.get("frontend", {})),
            output=OutputConfig(**raw.get("output", {})),
            processing=ProcessingConfig(**raw.get("processing", {})),
        )


# --------------------------------------------------------------------------- #
# DSP front-end
# --------------------------------------------------------------------------- #
class SpectralFeatureExtractor:
    """Stateless DSP front-end: one waveform -> one fixed-length feature vector.

    The extractor computes a single STFT and derives every requested feature
    group from it, then summarises each per-frame feature over time with the
    configured statistics (mean/std). The output dimensionality and column
    order are fully determined by the config and exposed via
    :attr:`feature_names`.

    Example:
        >>> fe = SpectralFeatureExtractor(FrontendConfig())
        >>> vec = fe.transform(waveform_16k_mono)
        >>> len(vec) == len(fe.feature_names)
        True
    """

    #: Internal registry: group name -> (column prefix, has per-channel index).
    _PREFIX = {
        "mfcc": ("mfcc", True),
        "delta": ("d1_", True),
        "delta2": ("d2_", True),
        "mel": ("mel", True),
        "contrast": ("contrast", True),
        "chroma": ("chroma", True),
        "centroid": ("centroid", False),
        "bandwidth": ("bandwidth", False),
        "rolloff": ("rolloff", False),
        "flatness": ("flatness", False),
        "zcr": ("zcr", False),
        "rms": ("rms", False),
    }

    def __init__(self, config: FrontendConfig | None = None) -> None:
        """Initialise the front-end with a :class:`FrontendConfig`."""
        self.cfg = config or FrontendConfig()
        unknown = set(self.cfg.groups) - set(self._PREFIX)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        #: Minimum #samples guaranteeing >= 9 frames so delta(width=9) is valid.
        self._min_samples = self.cfg.n_fft + 8 * self.cfg.hop_length
        self._channels = self._compute_channel_counts()

    # ----- public API ----------------------------------------------------- #
    @property
    def feature_names(self) -> list[str]:
        """Ordered list of output column names (length == feature dimension)."""
        names: list[str] = []
        for group in self.cfg.groups:
            prefix, indexed = self._PREFIX[group]
            n = self._channels[group]
            labels = [f"{prefix}{i}" for i in range(n)] if indexed else [prefix]
            for stat in self.cfg.aggregate:
                names.extend(f"{lab}_{stat}" for lab in labels)
        return names

    @property
    def n_features(self) -> int:
        """Total number of output features."""
        return len(self.feature_names)

    def transform(self, waveform: np.ndarray) -> np.ndarray:
        """Convert one mono waveform into a 1-D ``float32`` feature vector.

        Args:
            waveform: Mono audio at ``cfg.sample_rate`` (1-D float array). Short
                clips are zero-padded up to the minimum length required by the
                delta filters.

        Returns:
            A ``float32`` vector of length :attr:`n_features`, with NaN/inf
            replaced by zeros.
        """
        y = np.asarray(waveform, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < self._min_samples:
            y = np.pad(y, (0, self._min_samples - len(y)))

        power, mag, mel_db = self._spectra(y)
        blocks = [self._aggregate(self._group(g, y, power, mag, mel_db))
                  for g in self.cfg.groups]
        vec = np.concatenate(blocks).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- internals ------------------------------------------------------- #
    def _compute_channel_counts(self) -> dict[str, int]:
        """Resolve how many channels each enabled group produces."""
        counts = {
            "mfcc": self.cfg.n_mfcc, "delta": self.cfg.n_mfcc,
            "delta2": self.cfg.n_mfcc, "mel": self.cfg.n_mels,
            "contrast": 7, "chroma": 12,
            "centroid": 1, "bandwidth": 1, "rolloff": 1,
            "flatness": 1, "zcr": 1, "rms": 1,
        }
        return {g: counts[g] for g in self.cfg.groups}

    def _spectra(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the shared power/magnitude spectrograms and log-mel (dB)."""
        c = self.cfg
        power = np.abs(librosa.stft(y, n_fft=c.n_fft, hop_length=c.hop_length)) ** 2
        mag = np.sqrt(power)
        mel = librosa.feature.melspectrogram(
            S=power, sr=c.sample_rate, n_mels=c.n_mels, fmin=c.fmin, fmax=c.fmax)
        mel_db = librosa.power_to_db(mel)
        return power, mag, mel_db

    def _group(self, group: str, y: np.ndarray, power: np.ndarray,
               mag: np.ndarray, mel_db: np.ndarray) -> np.ndarray:
        """Return the per-frame feature matrix ``(channels, frames)`` of a group."""
        c = self.cfg
        if group == "mfcc":
            return librosa.feature.mfcc(S=mel_db, n_mfcc=c.n_mfcc)
        if group == "delta":
            return librosa.feature.delta(librosa.feature.mfcc(S=mel_db, n_mfcc=c.n_mfcc))
        if group == "delta2":
            return librosa.feature.delta(
                librosa.feature.mfcc(S=mel_db, n_mfcc=c.n_mfcc), order=2)
        if group == "mel":
            return mel_db
        if group == "contrast":
            return librosa.feature.spectral_contrast(S=mag, sr=c.sample_rate)
        if group == "chroma":
            return librosa.feature.chroma_stft(S=power, sr=c.sample_rate)
        if group == "centroid":
            return librosa.feature.spectral_centroid(S=mag, sr=c.sample_rate)
        if group == "bandwidth":
            return librosa.feature.spectral_bandwidth(S=mag, sr=c.sample_rate)
        if group == "rolloff":
            return librosa.feature.spectral_rolloff(S=mag, sr=c.sample_rate)
        if group == "flatness":
            return librosa.feature.spectral_flatness(S=mag)
        if group == "zcr":
            return librosa.feature.zero_crossing_rate(y, hop_length=c.hop_length)
        if group == "rms":
            return librosa.feature.rms(S=mag, frame_length=c.n_fft)
        raise ValueError(group)  # unreachable; validated in __init__

    def _aggregate(self, frames: np.ndarray) -> np.ndarray:
        """Aggregate a ``(channels, frames)`` matrix over time per statistic.

        Order matches :attr:`feature_names`: all channel means, then all stds.
        """
        out = []
        for stat in self.cfg.aggregate:
            if stat == "mean":
                out.append(frames.mean(axis=1))
            elif stat == "std":
                out.append(frames.std(axis=1))
            elif stat == "min":
                out.append(frames.min(axis=1))
            elif stat == "max":
                out.append(frames.max(axis=1))
            else:
                raise ValueError(f"Unknown aggregate stat: {stat}")
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
class DatasetFeatureExtractor:
    """Run :class:`SpectralFeatureExtractor` over a whole dataset.

    Loads the source dataset, extracts features for every clip in parallel,
    carries the configured metadata columns through, and persists the result.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        """Store config and build the shared front-end."""
        self.config = config
        self.frontend = SpectralFeatureExtractor(config.frontend)

    # ----- public API ----------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        """Extract features for the whole dataset and save them.

        Returns:
            The resulting feature table as a :class:`pandas.DataFrame`.
        """
        meta = self._load_metadata()
        n = len(meta)
        names = self.frontend.feature_names
        print(f"[extract] clips={n}  features={len(names)}  "
              f"source={self.config.dataset.source}:{self.config.dataset.repo_or_path}")

        chunk = self.config.processing.chunk_size
        ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
        results = Parallel(n_jobs=self.config.processing.n_jobs, verbose=5)(
            delayed(self._process_range)(lo, hi) for lo, hi in ranges)

        X = np.empty((n, len(names)), dtype=np.float32)
        for lo, arr in results:
            X[lo:lo + arr.shape[0]] = arr

        df = pd.DataFrame(X, columns=names)
        for col in self.config.dataset.keep_cols:
            if col in meta.columns:
                df[col] = meta[col].values
            else:
                print(f"[extract] WARNING: keep_col '{col}' not found in source")
        self._save(df)
        return df

    # ----- dataset loading ------------------------------------------------- #
    def _load_dataset(self):
        """Load the source HF dataset (Hub or local ``save_to_disk``)."""
        from datasets import Audio, load_dataset, load_from_disk

        d = self.config.dataset
        if d.source == "hf":
            ds = load_dataset(d.repo_or_path, d.name, split=d.split)
        elif d.source == "local":
            ds = load_from_disk(d.repo_or_path)
            if hasattr(ds, "keys"):           # DatasetDict -> pick split
                ds = ds[d.split]
        else:
            raise ValueError(f"dataset.source must be 'hf' or 'local', got {d.source!r}")
        # Disable decoding: we read bytes/path ourselves (no torchcodec dependency).
        if d.audio_col in ds.column_names:
            ds = ds.cast_column(d.audio_col, Audio(decode=False))
        return ds

    def _load_metadata(self) -> pd.DataFrame:
        """Load just the metadata columns (no audio) to learn the row count."""
        ds = self._load_dataset()
        cols = [c for c in self.config.dataset.keep_cols if c in ds.column_names]
        return ds.select_columns(cols).to_pandas() if cols else pd.DataFrame(index=range(len(ds)))

    def _load_waveform(self, cell: Any) -> np.ndarray:
        """Decode one audio cell to a mono waveform at the target sample rate.

        Handles the three shapes that show up in practice: a ``{bytes, path}``
        struct (HF ``Audio`` storage), a bare file-path string, or an already
        decoded ``{array, sampling_rate}`` dict.
        """
        sr_target = self.config.frontend.sample_rate
        if isinstance(cell, dict) and cell.get("array") is not None:
            y, sr = np.asarray(cell["array"], dtype=np.float32), cell["sampling_rate"]
        elif isinstance(cell, dict) and cell.get("bytes") is not None:
            y, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif isinstance(cell, dict) and cell.get("path"):
            y, sr = sf.read(cell["path"], dtype="float32")
        elif isinstance(cell, str):
            y, sr = sf.read(cell, dtype="float32")
        else:
            raise ValueError(f"Unsupported audio cell type: {type(cell)}")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        return y

    def _process_range(self, lo: int, hi: int) -> tuple[int, np.ndarray]:
        """Worker: extract features for clips ``[lo, hi)``. Reloads the dataset.

        Re-loading inside the worker avoids pickling a memory-mapped dataset
        across processes; ``load_from_disk``/cached ``load_dataset`` is cheap.
        """
        ds = self._load_dataset()
        audio_col = self.config.dataset.audio_col
        out = np.empty((hi - lo, self.frontend.n_features), dtype=np.float32)
        for k, i in enumerate(range(lo, hi)):
            out[k] = self.frontend.transform(self._load_waveform(ds[i][audio_col]))
        return lo, out

    # ----- saving ---------------------------------------------------------- #
    def _save(self, df: pd.DataFrame) -> None:
        """Persist the feature table locally or push it to the Hub."""
        out = self.config.output
        if out.sink == "local":
            Path(out.path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out.path, index=False)
            print(f"[extract] saved -> {out.path}  shape={df.shape}")
        elif out.sink == "hf":
            from datasets import Dataset
            Dataset.from_pandas(df, preserve_index=False).push_to_hub(
                out.hf_repo, private=out.private)
            print(f"[extract] pushed -> hf://{out.hf_repo}  shape={df.shape}")
        else:
            raise ValueError(f"output.sink must be 'local' or 'hf', got {out.sink!r}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Entry point: ``python -m utils.feature_extractor --config config.yaml``."""
    parser = argparse.ArgumentParser(description="Extract spectral features for KWS.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override worker count.")
    args = parser.parse_args()

    config = ExtractionConfig.from_yaml(args.config)
    if args.n_jobs is not None:
        config.processing.n_jobs = args.n_jobs
    DatasetFeatureExtractor(config).run()


if __name__ == "__main__":
    main()
