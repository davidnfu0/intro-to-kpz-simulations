"""
deposition_models.deposition_models
Author: David Felipe

This module provides implementations for simple 1D surface growth models,
including relaxation and ballistic deposition, along with a visualization tool.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def find_highest_point(col: np.array) -> int:
    """
    Return the highest occupied index in a given column.

    Parameters:
    col (np.array): A 1D array representing a column of the grid.

    Returns:
    int: Index of the highest non-zero entry, or -1 if the column is empty.
    """
    occupied = np.where(col > 0)[0]
    return occupied.max() if occupied.size > 0 else -1


def relax_deposition_step(grid: np.array, x: int) -> np.array:
    """
    Apply a single step of the relaxation deposition model.

    A particle drops at column `x` and may move left or right to minimize height.

    Parameters:
    grid (np.array): 2D array representing the surface.
    x (int): Column index where the particle initially falls.

    Returns:
    np.array: Updated grid after the deposition step.
    """
    y = find_highest_point(grid[:, x]) + 1

    while True:
        moved = False

        if x > 0:
            y_left = find_highest_point(grid[:, x - 1]) + 1
            if y - y_left >= 2:
                x -= 1
                y = y_left
                moved = True

        if moved:
            continue

        if x < grid.shape[1] - 1:
            y_right = find_highest_point(grid[:, x + 1]) + 1
            if y - y_right >= 2:
                x += 1
                y = y_right
                moved = True

        if not moved:
            break

    if 0 <= y < grid.shape[0]:
        grid[y, x] += 1

    return grid


def balistic_deposition_step(grid: np.array, x: int) -> np.array:
    """
    Apply a single step of the ballistic deposition model.

    A particle drops at column `x` and sticks to the highest neighboring column.

    Parameters:
    grid (np.array): 2D array representing the surface.
    x (int): Column index where the particle initially falls.

    Returns:
    np.array: Updated grid after the deposition step.
    """
    y = find_highest_point(grid[:, x]) + 1

    if x > 0 and x < grid.shape[1] - 1:
        y_right = find_highest_point(grid[:, x + 1])
        y_left = find_highest_point(grid[:, x - 1])
        y = max(y, y_left, y_right)

    elif x > 0:
        y_left = find_highest_point(grid[:, x - 1])
        y = max(y, y_left)

    elif x < grid.shape[1] - 1:
        y_right = find_highest_point(grid[:, x + 1])
        y = max(y, y_right)

    if 0 <= y < grid.shape[0]:
        grid[y, x] += 1

    return grid


def simulate_relaxation_deposition(grid: np.array, n_particles: int) -> np.array:
    """
    Simulate relaxation deposition for a number of particles.

    Parameters:
    grid (np.array): 2D grid representing the surface.
    n_particles (int): Number of particles to deposit.

    Returns:
    np.array: Grid after all particles have been deposited.
    """
    for _ in range(n_particles):
        x = np.random.randint(0, grid.shape[1])
        grid = relax_deposition_step(grid, x)
    return grid


def simulate_ballistic_deposition(grid: np.array, n_particles: int) -> np.array:
    """
    Simulate ballistic deposition for a number of particles.

    Parameters:
    grid (np.array): 2D grid representing the surface.
    n_particles (int): Number of particles to deposit.

    Returns:
    np.array: Grid after all particles have been deposited.
    """
    for _ in range(n_particles):
        x = np.random.randint(0, grid.shape[1])
        grid = balistic_deposition_step(grid, x)
    return grid


def create_animation_deposition_model(
    grid_size: tuple,
    n_particles: int,
    deposition_type: str,
    interval: int = 1000,
    color_quantity: int = 3,
    seed: int = None,
    often_parts: int = 1,
) -> animation.FuncAnimation:
    """
    Generate an animation of the deposition process.

    Parameters:
    grid_size (tuple): Shape of the grid (rows, columns).
    n_particles (int): Total number of particles to deposit.
    deposition_type (str): Type of model: 'relaxation' or 'ballistic'.
    interval (int): Delay between frames in milliseconds.
    color_quantity (int): Number of color levels to visualize time evolution.
    seed (int): Random seed for reproducibility.
    often_parts (int): Particles added per animation frame.

    Returns:
    animation.FuncAnimation: Animation of the simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros(grid_size, dtype=int)
    particles_count = 0
    n_particles_color = 0
    n_particles_per_color = n_particles // color_quantity + 1
    frames = n_particles // often_parts

    if deposition_type not in ["relaxation", "ballistic"]:
        raise ValueError("Invalid deposition type. Choose 'relaxation' or 'ballistic'.")

    title = f"{deposition_type.capitalize()} Deposition Model"
    fig, ax = plt.subplots(figsize=(7, 7))

    def animate(i):
        nonlocal grid, particles_count, n_particles_color

        for _ in range(often_parts):
            x = np.random.randint(0, grid.shape[1])
            if deposition_type == "relaxation":
                grid = relax_deposition_step(grid, x)
            else:
                grid = balistic_deposition_step(grid, x)

        ax.clear()
        ax.imshow(-1 * grid, cmap="bone", interpolation="none", origin="lower")
        ax.set_title(f"{title} - {particles_count} particles")
        ax.set_xlabel(r"$\mathbb{Z}$")
        ax.set_ylabel("height")

        particles_count += often_parts
        n_particles_color += often_parts

        if n_particles_color >= n_particles_per_color:
            n_particles_color = 0
            grid *= 2

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, repeat=True
    )
    return anim
