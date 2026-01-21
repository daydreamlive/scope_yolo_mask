"""YOLO26 person segmentation preprocessor plugin for Daydream Scope."""

from .plugin import YOLOMaskPlugin

plugin = YOLOMaskPlugin()

__all__ = ["plugin", "YOLOMaskPlugin"]
