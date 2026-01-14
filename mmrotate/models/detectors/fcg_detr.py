# Copyright (c) OpenMMLab. All rights reserved.
"""FCG-DETR: Foreground Confidence-Guided Detection Transformer.

This module implements the FCG-DETR detector for oriented object detection
in remote sensing images.
"""

from ..builder import ROTATED_DETECTORS
from .rotated_detr import RotatedDETR


@ROTATED_DETECTORS.register_module()
class FCGDETR(RotatedDETR):
    """FCG-DETR detector for oriented object detection.
    
    This detector extends RotatedDETR with foreground confidence guidance
    mechanism to improve detection performance on remote sensing images.
    """

    def __init__(self, *args, **kwargs):
        """Initialize FCG-DETR detector."""
        super(RotatedDETR, self).__init__(*args, **kwargs)
