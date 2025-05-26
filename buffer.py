"""Module for managing a fixed-size frame buffer using deque."""

from collections import deque
from typing import Any, TypeAlias

# Define a type alias for a frame, which is expected to be a NumPy array from OpenCV.
# Using Any for now as the exact type can be more complex (e.g., numpy.ndarray).
ImageFrame: TypeAlias = Any

# Define a type alias for the frame buffer itself.
FrameBuffer: TypeAlias = deque[ImageFrame]

def init_frame_buffer(size: int = 10) -> FrameBuffer:
    """
    Initializes a new frame buffer (a deque) of a specified fixed size.

    Args:
        size (int, optional): The maximum number of frames the buffer can hold.
                              Defaults to 10.

    Returns:
        FrameBuffer: An empty deque with a maximum length set to `size`.
    """
    if size <= 0:
        raise ValueError("Buffer size must be a positive integer.")
    return deque(maxlen=size)

def update_buffer(buffer: FrameBuffer, frame: ImageFrame) -> None:
    """
    Appends a new frame to the buffer.

    If the buffer is full (i.e., its length equals its `maxlen`),
    appending a new frame will automatically discard the oldest frame
    from the left end of the deque.

    Args:
        buffer (FrameBuffer): The frame buffer (deque) to update.
        frame (ImageFrame): The new frame to add to the buffer.
    """
    buffer.append(frame)

if __name__ == '__main__':
    print("--- Testing Frame Buffer ---    ")
    # Initialize a buffer of size 3
    try:
        frame_buf = init_frame_buffer(size=3)
        print(f"Initialized buffer: {list(frame_buf)}, Max size: {frame_buf.maxlen}")

        # Add frames
        frame1 = "Frame1" # Using strings for simplicity in this example
        update_buffer(frame_buf, frame1)
        print(f"After adding '{frame1}': {list(frame_buf)}")

        frame2 = "Frame2"
        update_buffer(frame_buf, frame2)
        print(f"After adding '{frame2}': {list(frame_buf)}")

        frame3 = "Frame3"
        update_buffer(frame_buf, frame3)
        print(f"After adding '{frame3}': {list(frame_buf)}")
        # Buffer should be full now: [Frame1, Frame2, Frame3]

        # Add another frame, which should pop the oldest (Frame1)
        frame4 = "Frame4"
        update_buffer(frame_buf, frame4)
        print(f"After adding '{frame4}': {list(frame_buf)}")
        # Buffer should now be: [Frame2, Frame3, Frame4]

        # Test with buffer of size 1
        frame_buf_small = init_frame_buffer(size=1)
        print(f"\\nInitialized small buffer: {list(frame_buf_small)}, Max size: {frame_buf_small.maxlen}")
        update_buffer(frame_buf_small, "SmallFrame1")
        print(f"After adding 'SmallFrame1': {list(frame_buf_small)}")
        update_buffer(frame_buf_small, "SmallFrame2")
        print(f"After adding 'SmallFrame2': {list(frame_buf_small)}")

        # Test invalid size
        print("\\nTesting invalid buffer size:")
        try:
            init_frame_buffer(size=0)
        except ValueError as e:
            print(f"Caught expected error for size=0: {e}")
        try:
            init_frame_buffer(size=-1)
        except ValueError as e:
            print(f"Caught expected error for size=-1: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during buffer testing: {e}")