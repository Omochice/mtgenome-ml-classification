import argparse
import json
from pathlib import Path

import yaml
from Bio import SeqIO
from mizlab_tools import calculate_coordinates
from mizlab_tools.calculate_coordinates import calc_coord
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate coordinates of graphed sequences",
    )
    parser.add_argument(
        "acc2taxon",
        help="The path of acc2taxon.json",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="The path of config.yml that has `mapping`",
    )
    parser.add_argument(
        "--weights",
        "-w",
        help="The path of weights.json(Optional)",
    )
    parser.add_argument(
        "--use_weights_match_taxon",
        action="store_true",
        help="Use weights that match taxon",
    )
    parser.add_argument(
        "--gbk_root",
        required=True,
        help="The path of directory that has genbank files",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="The path of directory output coordinates",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_false",
        help="Show progress bar",
    )
    args = parser.parse_args()

    with open(args.acc2taxon) as f:
        acc2taxon = json.load(f)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)
    else:
        weights = defaultdict(lambda: 1)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for acc, taxon in tqdm(acc2taxon.items(), disable=args.verbose):
        p = Path(args.gbk_root) / f"{acc}.gbk"
        if args.use_weights_match_taxon:
            weight = weights[taxon]
        else:
            weight = weights
        for record in SeqIO.parse(p, "genbank"):
            coordinates = calc_coord(
                record,
                config["mapping"],
                weights,
            )
            with (outdir / f"{acc}.json").open("w") as f:
                json.dump(coordinates, f)
