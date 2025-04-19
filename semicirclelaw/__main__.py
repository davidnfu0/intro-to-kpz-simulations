"""
semicirclelaw.__main__
Author: David Felipe

Main entry point to generate and optionally save an animation showing
the convergence of eigenvalue distributions to the Wigner semicircle law.
"""

import matplotlib.pyplot as plt
from numpy.random import randint
from .semicirclelaw import create_animation_semicircle_law
from utils.save_animation import save_animation

# Configuration constants
NS = [10, 20, 50, 100, 200, 500, 1000, 2000]
MATRIX_TYPE = "GUE"
SEED = 97  # Set to None to use a random seed
INTERVAL = 1000  # in milliseconds
FILENAME = "semicircle_law_animation.gif"  # Set to None to disable saving

if SEED is None:
    SEED = randint(0, 1000000)


def main():
    """
    Create the animation of the semicircle law and optionally save it.
    """
    anim = create_animation_semicircle_law(
        Ns=NS,
        matrix_type=MATRIX_TYPE,
        seed=SEED,
        interval=INTERVAL,
    )

    # Save before showing to avoid modifying the animation state
    if FILENAME is not None:
        save_animation(anim, FILENAME)

    plt.show()


if __name__ == "__main__":
    main()
