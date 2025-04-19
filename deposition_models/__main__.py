"""
deposition_models.__main__
author: David Felipe
"""

from .deposition_models import create_animation_deposition_model
import matplotlib.pyplot as plt
from utils.save_animation import save_animation
from numpy.random import randint

# Constants
SEED = 97  # If None, a random seed will be generated
N_PARTICLES = 10000
GRID_SIZE = (200, 200)
INTERVAL = 50
COLOR_QUANTITY = 4
OFTEN_PARTS = 100
TYPE_DEPOSITION = "ballistic"
FILENAME = "resources/ballistic_deposition_model_animation.gif"  # If is None, the animation will not be saved
if SEED is None:
    SEED = randint(0, 1000000)


def main():
    """
    Main function to create and save the animation the deposition models.
    """

    anim = create_animation_deposition_model(
        seed=SEED,
        n_particles=N_PARTICLES,
        grid_size=GRID_SIZE,
        interval=INTERVAL,
        color_quantity=COLOR_QUANTITY,
        often_parts=OFTEN_PARTS,
        deposition_type=TYPE_DEPOSITION,
    )

    plt.show()

    if FILENAME is not None:
        del anim
        anim = create_animation_deposition_model(
            seed=SEED,
            n_particles=N_PARTICLES,
            grid_size=GRID_SIZE,
            interval=INTERVAL,
            color_quantity=COLOR_QUANTITY,
            often_parts=OFTEN_PARTS,
            deposition_type=TYPE_DEPOSITION,
        )
        save_animation(anim, FILENAME)


if __name__ == "__main__":
    main()
