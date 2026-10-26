"""Unified public API for the wave simulation physics backends."""

from physics.core import (
    CppWaveSim2D,
    TorchWaveSim2D,
    TorchWaveSim3D,
    WaveSim3D,
    WaveSimulator,
    cuda_available,
    device_label,
    pick_device,
)

__all__ = [
    "CppWaveSim2D",
    "TorchWaveSim2D",
    "TorchWaveSim3D",
    "WaveSim3D",
    "WaveSimulator",
    "cuda_available",
    "device_label",
    "pick_device",
]
