"""
utils.save_animation
Author: David Felipe

Utility to save matplotlib animations to file.
"""

import os
from matplotlib.animation import PillowWriter, FFMpegWriter


def save_animation(anim, filename: str, fps: int = 30) -> None:
    """
    Save the animation to a file.

    Parameters:
    anim (FuncAnimation): The animation object.
    filename (str): Destination filename. Extension determines format (e.g., .gif, .mp4).
    fps (int): Frames per second for the saved animation.

    Returns:
    None
    """
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
        elif ext in [".mp4", ".mkv", ".avi"]:
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        anim.save(filename, writer=writer)
        print(f"✅ Animation successfully saved to '{filename}'")

    except Exception as e:
        print(f"❌ Failed to save animation: {e}")
