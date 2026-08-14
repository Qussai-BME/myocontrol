"""
simulator.py - Multi-channel EMG Signal Simulator
MyoControl Suite v0.3

Generates realistic synthetic EMG for:
- Demo without hardware
- Quick testing
- Tutorial / teaching purposes

Supports multiple gestures and multi-channel recording.
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GestureProfile:
    """Per-channel intensity profile for a gesture."""
    name: str
    intensities: List[float]  # one per channel
    fatigue_rate: float = 0.0  # how fast the muscle fatigues


# Predefined gesture profiles (4 channels: flexor, extensor, thenar, hypothenar)
DEFAULT_GESTURES = {
    'rest':      GestureProfile('rest',      [0.02, 0.02, 0.02, 0.02]),
    'flexion':   GestureProfile('flexion',   [0.30, 0.05, 0.08, 0.05]),
    'extension': GestureProfile('extension', [0.05, 0.30, 0.05, 0.08]),
    'pinch':     GestureProfile('pinch',     [0.10, 0.08, 0.25, 0.05]),
    'grip':      GestureProfile('grip',      [0.20, 0.18, 0.22, 0.15],
                                 fatigue_rate=0.01),
    'supination':GestureProfile('supination',[0.15, 0.20, 0.05, 0.10]),
}


class EMGSimulator:
    """Generate realistic synthetic multi-channel EMG."""

    def __init__(self,
                 fs: int = 2000,
                 n_channels: int = 4,
                 gestures: Optional[Dict[str, GestureProfile]] = None,
                 seed: Optional[int] = None):
        self.fs = fs
        self.n_channels = n_channels
        self.gestures = gestures or DEFAULT_GESTURES
        self.rng = np.random.default_rng(seed)

    def generate_contraction(self,
                             duration: float,
                             gesture: str = 'flexion',
                             intensity_scale: float = 1.0,
                             add_noise: bool = True,
                             add_powerline: bool = True) -> np.ndarray:
        """
        Generate multi-channel EMG for a single contraction.

        Parameters
        ----------
        duration : seconds
        gesture : key into self.gestures
        intensity_scale : multiplier on base intensity
        add_noise : add baseline sensor noise
        add_powerline : add 50Hz powerline interference

        Returns
        -------
        (n_samples, n_channels) array
        """
        if gesture not in self.gestures:
            raise ValueError(
                f"Unknown gesture '{gesture}'. "
                f"Available: {list(self.gestures.keys())}")

        profile = self.gestures[gesture]
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)

        # Per-channel intensity
        intensities = profile.intensities[:self.n_channels]
        while len(intensities) < self.n_channels:
            intensities.append(0.02)

        signal = np.zeros((n_samples, self.n_channels))

        for ch in range(self.n_channels):
            # Envelope: bell-shaped contraction + breathing modulation
            envelope = np.ones(n_samples) * intensities[ch] * intensity_scale
            # Bell-shaped onset/offset
            onset_samples = min(n_samples // 5, 100)
            envelope[:onset_samples] *= np.linspace(0, 1, onset_samples)
            envelope[-onset_samples:] *= np.linspace(1, 0, onset_samples)
            # Low-frequency breathing
            envelope *= 1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t)

            # Fatigue (RMS decline over time)
            if profile.fatigue_rate > 0:
                fatigue = 1.0 - profile.fatigue_rate * t / duration
                envelope *= np.maximum(fatigue, 0.3)

            # EMG = envelope × stochastic motor unit firing
            motor_unit_firing = self.rng.normal(0, 1, n_samples)
            # Add high-frequency MUAPs (motor unit action potentials)
            for _ in range(5):
                freq = self.rng.uniform(20, 80)
                phase = self.rng.uniform(0, 2 * np.pi)
                motor_unit_firing += 0.3 * np.sin(
                    2 * np.pi * freq * t + phase)

            signal[:, ch] = envelope * motor_unit_firing

            # Baseline noise
            if add_noise:
                signal[:, ch] += self.rng.normal(0, 0.01, n_samples)

            # Powerline interference (50 Hz)
            if add_powerline:
                signal[:, ch] += 0.02 * np.sin(2 * np.pi * 50 * t)

        return signal

    def generate_sequence(self,
                          gesture_sequence: List[Dict]) -> np.ndarray:
        """
        Generate a sequence of gestures with rest periods.

        gesture_sequence: list of {'gesture': str, 'duration': float}
        """
        segments = []
        # Initial rest
        segments.append(self.generate_contraction(1.0, 'rest'))
        for item in gesture_sequence:
            segments.append(
                self.generate_contraction(item['duration'], item['gesture']))
            # Rest between gestures
            segments.append(self.generate_contraction(0.5, 'rest'))
        return np.vstack(segments)

    def list_gestures(self) -> List[str]:
        return list(self.gestures.keys())
