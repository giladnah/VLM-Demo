"""
Unit tests for video_source.py.

Covers expected, edge, and failure cases for the VideoStream class and related functions, including:
- File and RTSP source handling
- Frame reading, looping, and property access
- Error handling for unopened sources

Uses pytest and a DummyCapture class to mock OpenCV behavior.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from video_source import VideoStream, get_video_stream

class DummyCapture:
    def __init__(self, frames=3, fps=30, width=640, height=480, is_opened=True):
        self.frames = frames
        self.read_calls = 0
        self.fps = fps
        self.width = width
        self.height = height
        self._is_opened = is_opened
        self.released = False
        self.is_file = False  # Set by test
    def isOpened(self):
        return self._is_opened
    def read(self):
        if self.read_calls < self.frames:
            self.read_calls += 1
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            if self.is_file:
                self.read_calls = 1  # Simulate looping
                return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return False, None
    def set(self, prop, value):
        pass
    def release(self):
        self.released = True
    def get(self, prop):
        if prop == 5:  # CAP_PROP_FPS
            return self.fps
        elif prop == 3:  # CAP_PROP_FRAME_WIDTH
            return self.width
        elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return self.height
        return 0

@patch('cv2.VideoCapture')
def test_videostream_file_source(mock_vc):
    dummy = DummyCapture()
    dummy.is_file = True
    mock_vc.return_value = dummy
    vs = get_video_stream('test.mp4')
    assert isinstance(vs, VideoStream)
    assert vs.is_file
    assert vs.get_fps() == 30
    assert vs.get_frame_width() == 640
    assert vs.get_frame_height() == 480
    ret, frame = vs.read()
    assert ret and frame.shape == (480, 640, 3)
    # Simulate end of file and looping
    dummy.read_calls = dummy.frames
    ret, frame = vs.read()
    assert ret and frame.shape == (480, 640, 3)
    vs.release()
    assert dummy.released

@patch('cv2.VideoCapture')
def test_videostream_rtsp_source(mock_vc):
    dummy = DummyCapture()
    mock_vc.return_value = dummy
    vs = get_video_stream('rtsp://user:pass@host/stream')
    assert isinstance(vs, VideoStream)
    assert not vs.is_file
    ret, frame = vs.read()
    assert ret
    vs.release()
    assert dummy.released

@patch('cv2.VideoCapture')
def test_videostream_not_opened(mock_vc):
    dummy = DummyCapture(is_opened=False)
    mock_vc.return_value = dummy
    with pytest.raises(ValueError):
        get_video_stream('badfile.mp4')