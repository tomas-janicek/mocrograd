"""Implements simple neural networks framework."""

from .modules import Module, Linear
from .models import MLP, MLPDigits, MLPDigitsBigger, MLPDigitsLonger
from .loss import (
    AccuracyFunction,
    LossFunction,
    cross_entropy_loss,
    calculate_accuracy,
)
from .optimizers import Optimizer, SGD
