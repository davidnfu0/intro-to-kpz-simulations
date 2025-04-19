"""
semicirclelaw.semicirclelaw
Author: David Felipe

This module generates random matrices from Gaussian ensembles (GOE and GUE)
and visualizes the convergence of their eigenvalue distributions to the
Wigner semicircle law through animation.
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib import animation

SQRT2 = np.sqrt(2)


def generate_goe_matrix(N: int, seed=None) -> np.ndarray:
    """
    Generate a matrix from the Gaussian Orthogonal Ensemble (GOE).

    Parameters:
    N (int): Size of the square matrix (NxN).
    seed (int, optional): Random seed for reproducibility.

    Returns:
    np.ndarray: A real symmetric GOE matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.random.normal(size=(N, N))
    A = (A + A.T) / SQRT2

    for i in range(N):
        A[i, i] = SQRT2 * np.random.normal()

    return A


def generate_gue_matrix(N: int, seed=None) -> np.ndarray:
    """
    Generate a matrix from the Gaussian Unitary Ensemble (GUE).

    Parameters:
    N (int): Size of the square matrix (NxN).
    seed (int, optional): Random seed for reproducibility.

    Returns:
    np.ndarray: A complex Hermitian GUE matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.zeros((N, N), dtype=np.complex_)
    for i in range(N):
        for j in range(i + 1, N):
            A[i, j] = (np.random.normal() + 1j * np.random.normal()) / SQRT2
            A[j, i] = np.conjugate(A[i, j])

    for i in range(N):
        A[i, i] = np.random.normal()

    return A


def wigner_semicircle_law(x: np.ndarray) -> np.ndarray:
    """
    Compute the Wigner semicircle law at the given points.

    Parameters:
    x (np.ndarray): Points where the semicircle law is evaluated.

    Returns:
    np.ndarray: Semicircle law density values at the given points.
    """
    x = x * (x >= -2) * (x <= 2) + 2 * (x < -2) + 2 * (x > 2)
    return (1 / (2 * np.pi)) * np.sqrt(4 - x**2)


def create_animation_semicircle_law(
    Ns: list, matrix_type: str, seed: int = None, interval: int = 1000
) -> animation.FuncAnimation:
    """
    Create an animation showing the convergence of eigenvalue histograms
    to the Wigner semicircle law for increasing matrix sizes.

    Parameters:
    Ns (list): List of matrix sizes to generate.
    matrix_type (str): Type of ensemble ('GOE' or 'GUE').
    seed (int, optional): Random seed for reproducibility.
    interval (int): Delay between frames in milliseconds.

    Returns:
    animation.FuncAnimation: Matplotlib animation object.
    """
    eigenvalues = []
    for N in Ns:
        if matrix_type == "GOE":
            matrix = generate_goe_matrix(N, seed)
        elif matrix_type == "GUE":
            matrix = generate_gue_matrix(N, seed)
        else:
            raise ValueError("Invalid matrix type. Use 'GOE' or 'GUE'.")
        eigenvalues.append(eigh(matrix)[0] / np.sqrt(N))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-2.5, 2.5, 1000)
    y = wigner_semicircle_law(x)

    def animate(i):
        ax.clear()
        N = Ns[i]

        # Plot the theoretical Wigner semicircle
        ax.plot(x, y, color="blue", linewidth=1.8, label="Semicircle Law")

        # Plot the histogram of eigenvalues
        hist, bins = np.histogram(eigenvalues[i], bins=int(np.sqrt(N)), density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ax.bar(
            bin_centers,
            hist,
            width=bins[1] - bins[0],
            alpha=0.5,
            color="red",
            label=f"Eigenvalues rescaled (N={N})",
        )

        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\sigma(\lambda)$")
        ax.set_title(f"Wigner Semicircle Law for {matrix_type} with N={N}")
        ax.legend(loc="upper left")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(Ns), interval=interval, repeat=True
    )

    return anim
