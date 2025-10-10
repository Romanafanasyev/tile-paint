from __future__ import annotations

import imageio.v2 as imageio
import numpy as np

from painter.io.files import ensure_dir_for, unique_path


class VideoRecorder:
    """
    Thin wrapper over imageio writer with unique output path and fixed FPS.
    """

    def __init__(self, output_path: str, fps: int) -> None:
        ensure_dir_for(output_path)
        self.path: str = unique_path(output_path)
        self._writer = imageio.get_writer(self.path, fps=fps)
        self.fps = int(fps)

    def append_frame(self, frame_u8: np.ndarray) -> None:
        self._writer.append_data(frame_u8)

    def append_canvas(self, canvas01: np.ndarray) -> None:
        frame = np.clip(canvas01 * 255.0, 0, 255).astype(np.uint8)
        self._writer.append_data(frame)

    def hold(self, frame_u8: np.ndarray, frames: int) -> None:
        n = max(0, int(frames))
        for _ in range(n):
            self._writer.append_data(frame_u8)

    def close(self) -> None:
        self._writer.close()
