"""Real-time / file audio preprocessing for KWS inference.

This module is the *inference-side* counterpart of
``utils.feature_extractor``. It turns a continuous audio source -- either a
file of arbitrary length or a live microphone stream -- into a sequence of
fixed-length feature vectors by sliding a window over the signal and applying
the **exact same** :class:`SpectralFeatureExtractor` used to build the training
data. Reusing the same front-end is what prevents training/serving skew.

By design this module contains **no model**. It only produces feature vectors;
you connect them to any fitted ``scikit-learn`` estimator / ``Pipeline``
downstream. This keeps the DSP front-end and the classifier cleanly decoupled
and lets you drop the preprocessor straight into a serving pipeline.

Offline (file) usage::

    from utils.feature_extractor import SpectralFeatureExtractor, FrontendConfig
    from utils.stream_preprocessor import AudioPreprocessor

    pre = AudioPreprocessor(SpectralFeatureExtractor(FrontendConfig()),
                            window_sec=1.0, hop_sec=0.1)
    timestamps, X = pre.process_file("clip.ogg")     # X: (n_windows, n_features)
    posteriors = fitted_pipeline.predict_proba(X)[:, 1]

Online (microphone) usage::

    for frame in pre.iter_stream():                  # needs PyAudio + PortAudio
        p = fitted_pipeline.predict_proba(frame.features[None])[0, 1]
        ...                                          # smooth / threshold / debounce
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf

from .feature_extractor import FrontendConfig, SpectralFeatureExtractor


@dataclass
class FrameFeatures:
    """One windowed feature vector produced by the preprocessor.

    Attributes:
        timestamp: Time in seconds (window centre for files, window end for the
            live stream) the vector corresponds to.
        features: The ``float32`` feature vector, ready for ``predict``.
    """

    timestamp: float
    features: np.ndarray


class AudioPreprocessor:
    """Slide a window over audio and emit feature vectors (no model inside).

    The preprocessor wraps a :class:`SpectralFeatureExtractor` and adds the
    windowing/streaming logic required to apply a clip-level front-end to
    continuous audio.

    Args:
        extractor: The shared DSP front-end (same config as training).
        window_sec: Analysis window length in seconds (~ one keyword length).
        hop_sec: Step between consecutive windows in seconds (lower = lower
            latency and finer time resolution, but more compute).
    """

    def __init__(self, extractor: SpectralFeatureExtractor | None = None,
                 window_sec: float = 1.0, hop_sec: float = 0.1) -> None:
        """Initialise with a front-end and window/hop sizes."""
        self.extractor = extractor or SpectralFeatureExtractor(FrontendConfig())
        self.sample_rate = self.extractor.cfg.sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window = int(round(window_sec * self.sample_rate))
        self.hop = int(round(hop_sec * self.sample_rate))

    # ----- offline (file) -------------------------------------------------- #
    def process_waveform(self, waveform: np.ndarray,
                         sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Window a whole waveform and extract a feature vector per window.

        Args:
            waveform: Audio samples (mono or multi-channel).
            sample_rate: Sample rate of ``waveform``; resampled if needed.

        Returns:
            ``(timestamps, X)`` where ``timestamps`` is a 1-D array of window
            centres (seconds) and ``X`` has shape ``(n_windows, n_features)``.
        """
        y = self._to_mono_target_sr(waveform, sample_rate)
        if len(y) < self.window:                 # too short -> single padded window
            return (np.array([len(y) / 2 / self.sample_rate], dtype=np.float32),
                    self.extractor.transform(y)[None, :])

        starts = range(0, len(y) - self.window + 1, self.hop)
        feats = np.stack([self.extractor.transform(y[s:s + self.window]) for s in starts])
        centres = np.array([(s + self.window / 2) / self.sample_rate for s in starts],
                           dtype=np.float32)
        return centres, feats

    def process_file(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load an audio file and run :meth:`process_waveform` on it."""
        y, sr = sf.read(path, dtype="float32")
        return self.process_waveform(y, sr)

    # ----- online (microphone) -------------------------------------------- #
    def iter_stream(self, device_index: int | None = None) -> Iterator[FrameFeatures]:
        """Yield feature vectors from a live microphone in real time.

        Maintains a ring buffer of the last ``window`` samples; every ``hop``
        samples it extracts one feature vector. Requires the optional
        ``PyAudio`` dependency (and a working PortAudio install).

        Args:
            device_index: Optional input device index (PyAudio enumeration).

        Yields:
            :class:`FrameFeatures` with the window-end timestamp.
        """
        try:
            import pyaudio
        except ImportError as exc:               # pragma: no cover - optional dep
            raise ImportError(
                "Live streaming needs PyAudio. Install it (and system PortAudio): "
                "`pip install pyaudio` + `apt-get install portaudio19-dev`."
            ) from exc

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate,
                         input=True, frames_per_buffer=self.hop,
                         input_device_index=device_index)
        ring = np.zeros(self.window, dtype=np.float32)
        n_seen = 0
        try:
            while True:
                chunk = np.frombuffer(
                    stream.read(self.hop, exception_on_overflow=False), dtype=np.float32)
                ring = np.roll(ring, -len(chunk))
                ring[-len(chunk):] = chunk
                n_seen += len(chunk)
                yield FrameFeatures(timestamp=n_seen / self.sample_rate,
                                    features=self.extractor.transform(ring))
        finally:                                  # always release the device
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ----- helpers --------------------------------------------------------- #
    def _to_mono_target_sr(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Down-mix to mono and resample to the front-end's sample rate."""
        y = np.asarray(waveform, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sample_rate != self.sample_rate:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=self.sample_rate)
        return y


__all__ = ["AudioPreprocessor", "FrameFeatures"]
