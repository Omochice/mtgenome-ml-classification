import argparse
import math

import numpy as np
from PIL import Image


def mix_monochromes(R: np.array, G: np.array, B: np.array) -> Image:
    stacked = np.stack((R, G, B), axis=2)
    return Image.fromarray(stacked)


def load_as_monochrome(filepath: str) -> np.array:
    return np.array(Image.open(filepath).convert("L"))


if __name__ == "__main__":
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

    args = parser.parse_args()

    images = [load_as_monochrome(f) for f in args.images]

    image = mix_monochromes(*images)
    # width, height = image.size
    # center_w, center_h = map(lambda x: math.ceil(x // 2) - 1, (width, height))
    # w_range = 2 - width % 2
    # h_range = 2 - height % 2
    # for dh in range(-1, h_range + 1):
    #     for dw in range(-1, w_range + 1):
    #         w = center_w + dw
    #         h = center_h + dh
    #         print(f"[{w}, {h}] => {image.getpixel((w, h))}")
    if args.show:
        image.show()
    else:
        image.save("./output.pnm")
