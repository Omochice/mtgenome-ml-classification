import argparse
import sys
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse arguments

    Args:

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "acc2taxon",
        help="The path to acc2taxon.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="The path to directory that have images",
    )
    parser.add_argument(
        "--group-name",
        required=True,
        help="Group name",
    )
    parser.add_argument(
        "--max-images",
        default=10,
        help="The number of images",
    )
    parser.add_argument(
        "--out",
        "-o",
        help="The path to output",
    )
    return parser.parse_args()


def make_alignmented_figure(images: List[np.array]) -> plt.figure:
    # Delete figure if exists already
    plt.clf()
    fig = plt.figure(figsize=(4 * len(images), 6))
    for i, image in enumerate(images, start=1):
        ax = fig.add_subplot(1, len(images), i)
        ax.imshow(image)
        # `ax.axis("off")` make disable border too.
        # on this case border is needed
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
    return fig


def load_image(filepath: str) -> np.array:
    """Load image

    Args:
        filepath (str): The path to source image

    Returns:
        np.array: Loaded image as np.array
    """
    return np.array(Image.open(filepath).convert("RGB"))


if __name__ == "__main__":
    args = parse_args()

    with open(args.acc2taxon) as f:
        d = json.load(f)

    accs = [acc for acc, group in d.items() if group == args.group_name][
        : args.max_images
    ]

    if len(accs) == 0:
        print(f"No species belong to {args.group_name}", file=sys.stderr)
        sys.exit(1)

    images = [load_image(str(args.image_dir / f"{acc}.png")) for acc in accs]

    fig = make_alignmented_figure(images)

    # save
    if not args.out:
        args.out = f"{args.group_name}.png"
    plt.savefig(args.out, bbox_inches="tight")
