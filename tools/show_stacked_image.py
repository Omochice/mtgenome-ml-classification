import argparse

import numpy as np
from PIL import Image


def mix_monochromes(R: np.array, G: np.array, B: np.array) -> Image:
    """Generate color image from three monochrome images.

    Args:
        R (np.array): The image as R channel
        G (np.array): The image as G channel
        B (np.array): The image as B channel

    Returns:
        Image: Mixed color image
    """
    stacked = np.stack((R, G, B), axis=2)
    return Image.fromarray(stacked)


def load_as_monochrome(filepath: str) -> np.array:
    """Load image as monochrome

    Args:
        filepath (str): The path to source image

    Returns:
        np.array: Loaded image as np.array
    """
    return np.array(Image.open(filepath).convert("L"))


def parse_args() -> argparse.Namespace:
    """parse_args

    Args:

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description="Generate color image from 3 monochrome images."
    )
    parser.add_argument(
        "--images",
        "-i",
        nargs=3,
        help="The 3 images that be source to mix. They will be assigned to R/G/B",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show mixed image.",
    )
    parser.add_argument(
        "--out",
        "-o",
        help="The path to outout",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images = [load_as_monochrome(f) for f in args.images]

    image = mix_monochromes(*images)
    if args.show:
        image.show()
    else:
        image.save(args.out)
