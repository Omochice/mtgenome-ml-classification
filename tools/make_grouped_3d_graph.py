import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


@dataclasses.dataclass
class Coordinate:
    """Coordinate class"""

    x: Iterable[float]
    y: Iterable[float]
    z: Iterable[float]
    name: str


def parse_args() -> argparse.Namespace:
    """Parse arguments

    Args:

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "acc2taxon",
        help="path to `acc2taxon.json`",
    )
    parser.add_argument(
        "--coordinate-root",
        required=True,
        help="path to coordinate root directory",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="group name ex) Mammalia",
    )
    parser
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="max number to plot species",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="",
        help="Path to output directory",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="show legends into image",
    )
    parser.add_argument(
        "--exec-show",
        action="store_true",
        help="execute show command (not saving)",
    )
    return parser.parse_args()


def initialize(ax, title: str = None):
    """initialize graph

    Args:
        ax: axes
        title (str): title string
    """
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        ax.set_title(title)


def plot(ax, coordinate: Coordinate):
    """plot.

    Args:
        ax: axes
        coordinate (Coordinate): coordinate
    """
    ax.plot(
        coordinate.x,
        coordinate.y,
        coordinate.z,
        label=coordinate.name,
    )


def collect_species(acc2taxon: Dict[str, str], group_name: str) -> Dict[str, List[str]]:
    """collect_species.

    Args:
        acc2taxon (Dict[str, str]): acc2taxon
        group_name (str): group_name like "Mammalia" or "All"

    Returns:
        Dict[str, List[str]]:
    """
    results = defaultdict(list)
    for k, v in acc2taxon.items():
        results[v].append(k)
    if group_name == "All":
        return results
    else:
        return {group_name: results[group_name]}


def main():
    args = parse_args()
    with open(args.acc2taxon) as f:
        d = json.load(f)
    collected = collect_species(d, group_name=args.group)
    for k in collected:
        collected[k] = collected[k][: min(len(collected[k]), args.max)]

    for taxon, accessions in collected.items():
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        initialize(ax, taxon)

        for accession in accessions:
            with (Path(args.coordinate_root) / f"{accession}.json").open() as f:
                data = json.load(f)
            coor = Coordinate(
                x=[d[0] for d in data],
                y=[d[1] for d in data],
                z=[d[2] for d in data],
                name=accession,
            )
            plot(ax, coor)
        if args.show_legend:
            ax.legend(bbox_to_anchor=(1.1, 1), loc="upper left")

        if args.exec_show:
            fig.show()
        else:
            fig.savefig((Path(args.out) / f"{taxon}.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
