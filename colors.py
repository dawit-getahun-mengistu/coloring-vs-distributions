import numpy as np
import matplotlib.pyplot as plt

import colorsys


def generate_colors(n, saturation=1.0, brightness=1.0):
    """
    Generate n colors equally spaced in hue, then convert them to RGB.

    Args:
        n (int): Number of colors to generate.
        saturation (float): Saturation value (0.0 to 1.0). Defaults to 1.0.
        brightness (float): Brightness value (0.0 to 1.0). Defaults to 1.0.

    Returns:
        list of tuple: A list of RGB color tuples.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Equally spaced hues in [0, 1)
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        colors.append(rgb)
    return colors


if __name__ == '__main__':
    colors = generate_colors(100)
    grid_size = int(np.ceil(np.sqrt(len(colors))))
    img = np.ones((grid_size * 32, grid_size * 32, 3))

    for idx, c in enumerate(colors):
        print("color: ", c)
        row = idx // grid_size
        col = idx % grid_size
        img[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32, :] = c

    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()
