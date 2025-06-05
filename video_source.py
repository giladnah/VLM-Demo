"""Module for handling video stream input and I-frame extraction."""

from typing import Any # OpenCV typically doesn't have precise type hints for VideoCapture
import cv2
import numpy as np # Add numpy import
from typing import TypeAlias # Add TypeAlias import

# OpenCV constants with fallback for linters
CAP_PROP_POS_FRAMES = getattr(cv2, 'CAP_PROP_POS_FRAMES', 1)
CAP_PROP_FPS = getattr(cv2, 'CAP_PROP_FPS', 5)
CAP_PROP_FRAME_WIDTH = getattr(cv2, 'CAP_PROP_FRAME_WIDTH', 3)
CAP_PROP_FRAME_HEIGHT = getattr(cv2, 'CAP_PROP_FRAME_HEIGHT', 4)

# Type alias for an image frame, expected to be a NumPy array from OpenCV.
ImageFrame: TypeAlias = np.ndarray

class VideoStream:
    """Represents a video stream, encapsulating the OpenCV VideoCapture object."""
    def __init__(self, capture: 'cv2.VideoCapture', source_uri: str): # type: ignore
        """
        Initializes the VideoStream.

        Args:
            capture (cv2.VideoCapture): The OpenCV video capture object.
            source_uri (str): The URI of the video source.
        """
        if not capture.isOpened():
            raise ValueError(f"Could not open video stream from {source_uri}")
        self.capture = capture
        self.source_uri = source_uri
        self.is_file = not source_uri.lower().startswith("rtsp://")

    def read(self) -> tuple[bool, Any | None]:
        """
        Reads a frame from the video stream. If the source is a file and the end is reached, loops to the start.
        Returns:
            tuple[bool, Any | None]: A tuple containing a boolean indicating success
                                     and the frame (numpy array) if successful, None otherwise.
        """
        ret, frame = self.capture.read()
        if not ret and self.is_file:
            # Loop: seek to start and try again
            self.capture.set(CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
        return ret, frame

    def release(self) -> None:
        """Releases the video capture object."""
        self.capture.release()

    def get_fps(self) -> float:
        """
        Gets the frames per second (FPS) of the video stream.

        Returns:
            float: The FPS of the video stream.
        """
        return self.capture.get(CAP_PROP_FPS)

    def get_frame_width(self) -> int:
        """
        Gets the width of the frames in the video stream.

        Returns:
            int: The width of the frames.
        """
        return int(self.capture.get(CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self) -> int:
        """
        Gets the height of the frames in the video stream.

        Returns:
            int: The height of the frames.
        """
        return int(self.capture.get(CAP_PROP_FRAME_HEIGHT))


def get_video_stream(source_uri: str) -> VideoStream:
    """
    Connects to a video source (RTSP stream, local file path, or /dev/video* device) and returns a VideoStream object.

    Args:
        source_uri (str): The URI of the video source.
                          Can be an RTSP URL (e.g., "rtsp://user:pass@ip_address:port/path"),
                          a local file path (e.g., "/path/to/video.mp4"),
                          or a webcam device (e.g., "/dev/video0").

    Returns:
        VideoStream: An object representing the video stream.

    Raises:
        ValueError: If the video stream cannot be opened.
    """
    # Handle /dev/video* as webcam device (OpenCV expects int index or string path)
    if source_uri.startswith("/dev/video"):
        cap = cv2.VideoCapture(source_uri) # type: ignore[attr-defined]
    else:
        cap = cv2.VideoCapture(source_uri) # type: ignore[attr-defined]
    # Set FPS to 30 for all sources
    cap.set(CAP_PROP_FPS, 30)
    return VideoStream(capture=cap, source_uri=source_uri)


def extract_i_frame(stream: VideoStream) -> Any | None:
    """
    Extracts an I-frame (keyframe) from the video stream.
    This is a simplified version that reads the next available frame.
    True I-frame detection would require more complex parsing of the video stream
    or relying on camera/encoder properties if available.

    For now, this function will just read the next frame.
    Actual I-frame property is cv2.CAP_PROP_POS_FRAMES, but we need to analyze
    the frame type, or assume the camera sends I-frames periodically.

    Args:
        stream (VideoStream): The video stream to extract the I-frame from.

    Returns:
        Any | None: The extracted I-frame (as a numpy array) if successful, otherwise None.
                     The type is Any because OpenCV frames are numpy arrays.
    """
    # Reason: For simplicity in this initial version, we read the next frame.
    # True I-frame extraction is more complex and depends on the video codec and stream properties.
    # Some cameras might expose a property like 'IFRAME' or similar, but it's not standard in OpenCV.
    # Another approach is to check the frame type using ffprobe or similar tools,
    # or if we know the GOP (Group of Pictures) structure.
    ret, frame = stream.read()
    if not ret:
        return None
    return frame

if __name__ == '__main__':
    # Example Usage (requires a video file or accessible RTSP stream)
    # Replace with your video source
    TEST_SOURCE_FILE = "test_video.mp4" # Create a dummy file for testing if needed
    TEST_RTSP_STREAM = "rtsp://your_rtsp_stream_url"

    # Create a dummy video file for testing if it doesn't exist
    try:
        with open(TEST_SOURCE_FILE, "rb") as f:
            pass
        print(f"'{TEST_SOURCE_FILE}' found.")
    except FileNotFoundError:
        print(f"Creating dummy '{TEST_SOURCE_FILE}' for testing.")
        # Create a small dummy mp4 file for basic testing
        # This requires ffmpeg to be installed and in PATH
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=1280x720:rate=1", TEST_SOURCE_FILE],
                check=True, capture_output=True
            )
            print(f"Dummy video '{TEST_SOURCE_FILE}' created successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Could not create dummy video file using ffmpeg: {e}")
            print("Please ensure ffmpeg is installed and in your PATH, or provide a valid video file.")
            TEST_SOURCE_FILE = None # Disable file test if creation fails

    if TEST_SOURCE_FILE:
        print(f"\\n--- Testing with local file: {TEST_SOURCE_FILE} ---")
        try:
            video_file_stream = get_video_stream(TEST_SOURCE_FILE)
            print(f"Successfully opened video file. FPS: {video_file_stream.get_fps()}")
            i_frame_from_file = extract_i_frame(video_file_stream)
            if i_frame_from_file is not None:
                print(f"Extracted I-frame from file. Shape: {i_frame_from_file.shape}")
                cv2.imwrite("i_frame_from_file.jpg", i_frame_from_file) # type: ignore[attr-defined]
                print("Saved extracted frame as 'i_frame_from_file.jpg'")
            else:
                print("Failed to extract I-frame from file.")
            video_file_stream.release()
        except ValueError as e:
            print(f"Error with local file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with local file test: {e}")

    # print(f"\\n--- Testing with RTSP stream: {TEST_RTSP_STREAM} ---")
    # print("Note: This test will likely fail if a valid RTSP stream is not available at the URL.")
    # try:
    #     rtsp_stream = get_video_stream(TEST_RTSP_STREAM)
    #     print(f"Successfully opened RTSP stream. FPS: {rtsp_stream.get_fps()}")
    #     i_frame_from_rtsp = extract_i_frame(rtsp_stream)
    #     if i_frame_from_rtsp is not None:
    #         print(f"Extracted I-frame from RTSP. Shape: {i_frame_from_rtsp.shape}")
    #         cv2.imwrite("i_frame_from_rtsp.jpg", i_frame_from_rtsp)
    #         print("Saved extracted frame as 'i_frame_from_rtsp.jpg'")
    #     else:
    #         print("Failed to extract I-frame from RTSP stream (may be normal if stream is slow to start or unavailable).")
    #     rtsp_stream.release()
    # except ValueError as e:
    #     print(f"Error with RTSP stream: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred with RTSP test: {e}")

    print("\\n--- Example of how to use VideoStream directly ---")
    if TEST_SOURCE_FILE:
        try:
            raw_capture = cv2.VideoCapture(TEST_SOURCE_FILE) # type: ignore[attr-defined]
            if raw_capture.isOpened():
                vs = VideoStream(raw_capture, TEST_SOURCE_FILE)
                print(f"VideoStream created. FPS: {vs.get_fps()}, Resolution: {vs.get_frame_width()}x{vs.get_frame_height()}")
                ret, frame = vs.read()
                if ret and frame is not None:
                    print(f"Read a frame. Shape: {frame.shape}")
                else:
                    print("Could not read a frame using VideoStream.read()")
                vs.release()
            else:
                print(f"cv2.VideoCapture could not open {TEST_SOURCE_FILE}")
        except Exception as e:
            print(f"Error in direct VideoStream example: {e}")
    else:
        print("Skipping direct VideoStream example as test file is not available.")