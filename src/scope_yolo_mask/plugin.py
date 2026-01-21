"""Plugin class for Daydream Scope integration."""

import logging

from scope.core.plugins import hookimpl

from .pipeline import YOLOMaskPipeline

logger = logging.getLogger(__name__)


class YOLOMaskPlugin:
    """Scope plugin that provides YOLO26 person segmentation."""

    @hookimpl
    def register_pipelines(self, register):
        """Register the YOLO mask pipeline."""
        register(YOLOMaskPipeline)
        logger.info("Registered YOLO mask pipeline (YOLO26-seg)")
