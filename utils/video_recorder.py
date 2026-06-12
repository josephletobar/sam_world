from pathlib import Path

import cv2


class VideoRecorder:
    def __init__(self, output_path, fps=20, codec="mp4v"):
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()

    def write(self, frame):
        if frame is None:
            return

        height, width = frame.shape[:2]

        if self.writer is None:
            self.frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size
            )

        if (width, height) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.release()
