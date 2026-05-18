"""geomodel.training — Training utilities, losses, and trainer."""
from .geometry import GeometryLoss
from .trainer import Trainer
__all__ = ["GeometryLoss", "Trainer"]
