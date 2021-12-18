import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from Bio import SeqIO


@dataclasses.dataclass
class Coordinate:
    """Coordinate and its name"""

    name: str
    x: List[float] = dataclasses.field(default_factory=list)
    y: List[float] = dataclasses.field(default_factory=list)
    z: List[float] = dataclasses.field(default_factory=list)  # this is optional


def parse_args() -> argparse.Namespace:
    """parse_args.

    Args:

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description="Make grouped graph image. If specity group that has many species, use only 10 species."
    )
    parser.add_argument(
        "acc2taxon",
        help="The path to acc2taxon",
    )
    parser.add_argument(
        "--coor-root",
        "-c",
        required=True,
        type=Path,
        help="The path to directory that has coordinats",
    )
    parser.add_argument(
        "--gbk-root",
        required=True,
        type=Path,
        help="The path to direcotry that has genbank files",
    )
    parser.add_argument(
        "--group-name",
        required=True,
        help="The name for group",
    )
    parser.add_argument(
        "--max-species",
        default=10,
        type=int,
        help="The number of max plot, default: 10",
    )
    parser.add_argument(
        "--threeD",
        action="store_true",
        help="Generate 3D image",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="The path to save figure. If missing this, save into {group-name}.png",
    )

    return parser.parse_args()


def extract_species(group_name: str, acc2taxon: Dict[str, str]) -> List[str]:
    """Extract accession from acc2taxon

    Args:
        group_name (str): The taxon name like "Mammalia"
        acc2taxon (Dict[str, str]): acc2taxon

    Returns:
        List[str]:
    """
    return [k for k, v in acc2taxon.items() if v == group_name]


def make_conparison(
    coordinates: List[Coordinate],
    is_3d: bool = False,
    title: str = None,
) -> plt.figure:
    """Generate figure with some coordinates

    Args:
        coordinates (List[Coordinate]): The list of Coordinate
        is_3d (bool): Whether make figure 2d or 3d, if specify this, Coordinates must have z coordinates
        title (str): The string to use as title

    Returns:
        plt.figure:
    """
    # Delete figure if exist already
    plt.clf()

    fig = plt.figure(figsize=(10, 10))
    if is_3d:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        ax = fig.add_subplot(1, 1, 1)

    for coor in coordinates:
        if is_3d:
            ax.plot(coor.x, coor.y, coor.z, label=coor.name)
        else:
            ax.plot(coor.x, coor.y, label=coor.name)

    ax.legend(prop={"style": "italic"})
    if title is not None:
        # use str + str. because using f"" is confusable if use {} in string.
        ax.set_title("$\\it{" + title + "}$")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # setting for each
    if args.threeD:
        # it looks difficult to set aspect equal in 3d figure
        ax.set_zlabel("Z-axis")
    else:
        ax.set_aspect("equal")
    return fig


if __name__ == "__main__":
    args = parse_args()
    with open(args.acc2taxon) as f:
        acc2taxon = json.load(f)

    species = extract_species(args.group_name, acc2taxon)
    # If miss group name, exit with error code
    if len(species) == 0:
        print(f"No species belong to {args.group_name}", file=sys.stderr)
        sys.exit(1)

    # get name and coordinates
    coordinates = []
    for acc in species[:10]:
        with (args.coor_root / f"{acc}.json").open() as f:
            c = json.load(f)
        for record in SeqIO.parse(args.gbk_root / f"{acc}.gbk", "genbank"):
            if args.threeD:
                coordinates.append(
                    Coordinate(
                        name=record.annotations["organism"],
                        x=list(map(lambda x: x[0], c)),
                        y=list(map(lambda x: x[1], c)),
                        z=list(map(lambda x: x[2], c)),
                    )
                )
            else:
                coordinates.append(
                    Coordinate(
                        name=record.annotations["organism"],
                        x=list(map(lambda x: x[0], c)),
                        y=list(map(lambda x: x[1], c)),
                    )
                )
    fig = make_conparison(coordinates, is_3d=args.threeD, title=args.group_name)

    # save
    if args.out is None:
        args.out = f"{args.group_name}.png"
    plt.savefig(args.out, bbox_inches="tight")
