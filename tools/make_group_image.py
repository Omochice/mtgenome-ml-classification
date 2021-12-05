import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from Bio import SeqIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make grouped graph image. If specity group that has many species, use only 10 species."
    )
    parser.add_argument(
        "acc2taxon",
        help="The path to acc2taxon",
    )
    parser.add_argument(
        "--coor_root",
        "-c",
        required=True,
        type=Path,
        help="The path to directory that has coordinats",
    )
    parser.add_argument(
        "--gbk_root",
        required=True,
        type=Path,
        help="The path to direcotry that has genbank files",
    )
    parser.add_argument(
        "--group_name",
        help="The name for group",
    )
    parser.add_argument(
        "--max_species",
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
        help="The path to save figure. If missing this, save into {group_name}.png",
    )

    args = parser.parse_args()

    with open(args.acc2taxon) as f:
        acc2taxon = json.load(f)

    species = [k for k, v in acc2taxon.items() if v == args.group_name]
    # If miss group name, return error
    if len(species) == 0:
        print(f"No species belong to {args.group_name}", file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(10, 10))
    if args.threeD:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        ax = fig.add_subplot(1, 1, 1)

    # plot
    for acc in species[:10]:
        with (args.coor_root / f"{acc}.json").open() as f:
            coordinates = json.load(f)
        for record in SeqIO.parse(args.gbk_root / f"{acc}.gbk", "genbank"):
            name = record.annotations["organism"]
        x = list(map(lambda x: x[0], coordinates))
        y = list(map(lambda x: x[1], coordinates))
        z = list(map(lambda x: x[2], coordinates))
        if args.threeD:
            ax.plot(x, y, z, label=name)
        else:
            ax.plot(x, y, label=name)

    ax.legend(prop={"style": "italic"})
    ax.set_title("$\\it{" + args.group_name + "}$")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # setting for each
    if args.threeD:
        ax.set_zlabel("Z-axis")
    else:
        ax.set_aspect("equal")

    # save
    if args.out is None:
        args.out = f"{args.group_name}.png"
    plt.savefig(args.out, bbox_inches="tight")
