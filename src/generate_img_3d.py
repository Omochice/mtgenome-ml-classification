import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib.collections import LineCollection
from tqdm import tqdm

from generate_coordinates import center_of_gravity

matplotlib.use("Agg")

# dont format

Coordinate = Union[int, float]
gradations = [str(i * (1 / 256)) for i in range(0, 256)]
gradations.reverse()


def format_graph_img(
    ax: matplotlib.axes._axes.Axes, xmin: float, xmax: float, ymin: float, ymax: float
) -> None:
    """縦横比を揃えたりする

    Args:
        fig (matplotlib.figure.Figure): グラフ画像

    Returns:
        matplotlib.figure.Figure: [description]
    """
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def plot(
    x: Iterable[Coordinate],
    y: Iterable[Coordinate],
    gradation: bool,
    save: bool,
    dst: Path,
    figsize: Tuple[float, float],
    dpi: int,
    linewidth: float = 0.1,
    gravity: bool = False,
) -> None:
    """plot tight graph.

    Args:
        x (Iterable[Coordinate]): x
        y (Iterable[Coordinate]): y
        gradation (bool): make head of line light color, tail dark color
        save (bool): save figure
        dst (Path): destination if save figure
        linewidth (float): width of plotted line
        gravity (bool): move gravity point to center of graph
    """
    tupled_x = tuple(x)
    tupled_y = tuple(y)
    assert len(tupled_x) == len(tupled_y)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if gradation:
        n_coords = len(tupled_x)
        n = n_coords / 256
        colors = [gradations[int(i // n)] for i in range(n_coords)]
        lines = [
            [(x1, y1), (x2, y2)]
            for x1, y1, x2, y2 in zip(tupled_x, tupled_y, tupled_x[1:], tupled_y[1:])
        ]
        lc = LineCollection(lines, colors=colors, linewidths=[linewidth] * n_coords)
        ax.add_collection(lc)
    else:
        ax.plot(tupled_x, tupled_y, color="Black", lw=linewidth)

    if gravity:
        gravity_x, gravity_y = center_of_gravity(tupled_x), center_of_gravity(tupled_y)
        xrange = max(max(tupled_x) - gravity_x, gravity_x - min(tupled_x))
        yrange = max(max(tupled_y) - gravity_y, gravity_y - min(tupled_y))
        (xmin, xmax, ymin, ymax) = (
            gravity_x - xrange,
            gravity_x + xrange,
            gravity_y - yrange,
            gravity_y + yrange,
        )
    else:
        (xmin, xmax, ymin, ymax) = (
            min(tupled_x),
            max(tupled_x),
            min(tupled_y),
            min(tupled_y),
        )
    format_graph_img(ax, xmin, xmax, ymin, ymax)
    if save:
        plt.savefig(dst)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images",
    )
    parser.add_argument(
        "coordinates",
        nargs="+",
        help="Files of coordinates",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="the path to config.yml",
    )
    parser.add_argument(
        "--gradation",
        action="store_true",
        help="Make head of line light color, tail dark color",
    )
    parser.add_argument(
        "--move_gravity",
        action="store_true",
        help="Match gravity point of line and center of graph",
    )
    parser.add_argument(
        "--out",
        "-o",
        help="The path of directory for output img",
    )
    parser.add_argument(
        "--verbose",
        action="store_false",
        help="Show progress bar",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    pix_a_side = config["graph_pix"]
    dpi = 100
    figsize = (pix_a_side / dpi, pix_a_side / dpi)
    graph_dst = Path(args.out)
    graph_dst.mkdir(parents=True, exist_ok=True)

    for coor_path in tqdm(
        map(lambda x: Path(x), args.coordinates),
        disable=args.verbose,
    ):
        with coor_path.open() as f:
            coords = json.load(f)
        accession = coor_path.stem
        for dim1, dim2 in ((0, 1), (1, 2), (2, 0)):
            # may fast to use numpy like coords[:, :, 1] ...
            dims1 = [c[dim1] for c in coords]
            dims2 = [c[dim2] for c in coords]
            dst = graph_dst / f"{accession}_{dim1}.png"
            plot(
                dims1,
                dims2,
                gradation=args.gradation,
                save=True,
                dst=dst,
                figsize=figsize,
                dpi=dpi,
                gravity=args.move_gravity,
            )
