import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from buffer import init_frame_buffer, update_buffer

def test_init_frame_buffer_valid():
    buf = init_frame_buffer(3)
    assert len(buf) == 0
    assert buf.maxlen == 3

def test_init_frame_buffer_invalid():
    with pytest.raises(ValueError):
        init_frame_buffer(0)
    with pytest.raises(ValueError):
        init_frame_buffer(-5)

def test_update_buffer_and_overflow():
    buf = init_frame_buffer(2)
    update_buffer(buf, 'frame1')
    update_buffer(buf, 'frame2')
    assert list(buf) == ['frame1', 'frame2']
    update_buffer(buf, 'frame3')
    assert list(buf) == ['frame2', 'frame3']  # Oldest dropped

def test_update_buffer_type_safety():
    buf = init_frame_buffer(2)
    update_buffer(buf, 123)  # Not a frame, but allowed by type
    update_buffer(buf, None)
    assert list(buf) == [123, None]