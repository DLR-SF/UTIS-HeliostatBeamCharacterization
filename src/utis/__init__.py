"""UTIS: UNet-based heliostat beam characterization for the camera-target method.

Reference: M. Kuhl et al., "In-situ UNet-based heliostat beam characterization
method for precise flux calculation using the camera-target method",
Solar Energy 279 (2024) 112811, https://doi.org/10.1016/j.solener.2024.112811
"""

from utis.flux import absolute_flux
from utis.inference import FluxPredictor, Prediction, load_model
from utis.model import UNet3Plus
from utis.targets import MFT, STJ_LOWER, STJ_UPPER, TARGETS, get_target

__version__ = "2.0.0"

__all__ = [
    "MFT",
    "STJ_LOWER",
    "STJ_UPPER",
    "TARGETS",
    "FluxPredictor",
    "Prediction",
    "UNet3Plus",
    "absolute_flux",
    "get_target",
    "load_model",
]
