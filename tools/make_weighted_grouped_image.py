import argparse
import collections
import itertools
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Union

import matplotlib.pyplot as plt
from Bio import SeqIO, SeqRecord
from make_group_image import Coordinate, make_conparison
from mizlab_tools import gbk_utils


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
        "--gbk-dir",
        type=Path,
        help="The path to directory that has gbk files",
    )
    parser.add_argument(
        "--apply-weights",
        action="store_true",
        help="Whether apply weights",
    )
    parser.add_argument(
        "--group-name",
        required=True,
        help="Use taxon name",
    )
    parser.add_argument(
        "--max-species",
        default=10,
        help="The number use species",
    )
    parser.add_argument(
        "--limits",
        nargs=4,
        type=float,
        help="[Optional] specify plot range (xmin, xmax, ymin, ymax)",
    )
    parser.add_argument(
        "--out",
        "-o",
        help="The path to output",
    )
    return parser.parse_args()


def calculate_weights(sequences: Iterable[str]):
    # count | convert freq | convert selfinfo
    return freq_to_selfinfo(
        count_to_freq(
            count_triplets(sequences),
        )
    )


def count_triplets(sequences: Iterable[str]) -> collections.Counter:
    counter = collections.Counter()
    for seq in sequences:
        seq = re.sub("[^atgcATGC]", "", str(record.seq))
        counter += collections.Counter(gbk_utils.window_search(seq, 3))
    return counter


def count_to_freq(
    counter: Union[Dict[str, int], collections.Counter], k: str = "ATGC"
) -> Dict[str, float]:
    frequencies = {}
    for twins in itertools.product(k, k):
        triplets = ["".join(twins) + last for last in k]
        sub_frequencies = [counter[triplet] for triplet in triplets]
        for triplet, freq in zip(triplets, sub_frequencies):
            frequencies[triplet] = freq / sum(sub_frequencies)
    return frequencies


def freq_to_selfinfo(freqencies: Dict[str, float]) -> Dict[str, float]:
    return {triplet: -1 * math.log2(freq) for triplet, freq in freqencies.items()}


if __name__ == "__main__":
    args = parse_args()

    # get use accessions
    with open(args.acc2taxon) as f:
        accs = [acc for acc, group in json.load(f).items() if group == args.group_name][
            : args.max_species
        ]

    # loading
    name2seq = {}
    for acc in accs:
        for record in SeqIO.parse(args.gbk_dir / f"{acc}.gbk", "genbank"):
            name2seq[record.annotations["organism"]] = re.sub(
                "[^atgcATGC]", "", str(record.seq)
            )

    # calc weight
    if args.apply_weights:
        weights = calculate_weights(name2seq.values())
    else:
        weights = {}

    # get coordinates
    mapping = {
        "A": [1, 1],
        "T": [1, -1],
        "G": [-1, -1],
        "C": [-1, 1],
    }
    coordinates = []
    for name, seq in name2seq.items():
        coor = [[0, 0]]
        for some_bases in gbk_utils.window_search(seq, 3, overhang="before"):
            now = [
                coor[-1][i]
                + (mapping[some_bases[-1]][i] * weights.get(some_bases, 1.0))
                for i in range(2)
            ]
            coor.append(now)
        coordinates.append(
            Coordinate(
                name=name,
                x=[c[0] for c in coor],
                y=[c[1] for c in coor],
            )
        )

    # plot
    if args.apply_weights:
        title = args.group_name + " (weighted)"
    else:
        title = args.group_name
    if args.limits:
        limits = {
            "xlim": (args.limits[0], args.limits[1]),
            "ylim": (args.limits[2], args.limits[3]),
        }
    else:
        limits = None

    fig = make_conparison(coordinates, is_3d=False, title=title, limits=limits)

    # save
    if not args.out:
        args.out = re.sub(r"\s", "-", re.sub(r"[\(\)]", "", title)) + ".png"
    plt.savefig(args.out, bbox_inches="tight")
