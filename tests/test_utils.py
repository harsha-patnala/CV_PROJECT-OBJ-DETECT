"""
tests/test_utils.py — Unit tests for utility functions.

Run with:
    ./venv/bin/python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from utils import (
    bgr_to_rgb,
    crop_bbox,
    draw_detection,
    frame_to_timestamp,
    get_box_colour,
    resize_frame,
    seconds_to_timestamp,
)


class TestFrameHelpers:
    def test_bgr_to_rgb_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = bgr_to_rgb(img)
        assert out.shape == img.shape

    def test_resize_frame_no_op_when_small(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = resize_frame(img, max_width=300)
        assert out.shape == img.shape

    def test_resize_frame_reduces_large(self):
        img = np.zeros((720, 1920, 3), dtype=np.uint8)
        out = resize_frame(img, max_width=960)
        assert out.shape[1] == 960

    def test_crop_bbox_valid(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(img, 10, 10, 50, 50, pad=0)
        assert crop is not None
        assert crop.shape == (40, 40, 3)

    def test_crop_bbox_out_of_bounds(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(img, 90, 90, 110, 110, pad=0)
        assert crop is not None  # Clipped, not None

    def test_crop_bbox_zero_area(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(img, 50, 50, 50, 50, pad=0)
        assert crop is None

    def test_get_box_colour_returns_tuple(self):
        c = get_box_colour(0)
        assert isinstance(c, tuple)
        assert len(c) == 3

    def test_get_box_colour_wraps(self):
        # Should not raise for any int
        for i in range(100):
            c = get_box_colour(i)
            assert len(c) == 3

    def test_draw_detection_returns_frame(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        out = draw_detection(img, 10, 10, 100, 100, "car #1 0.85", (0, 200, 80))
        assert out is not None
        assert out.shape == img.shape
