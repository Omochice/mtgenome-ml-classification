import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from Bio import SeqIO, SeqRecord
from mizlab_tools import gbk_utils
from tqdm import tqdm


def convert_freq_to_self_information(counter: Counter) -> Dict[str, float]:
    return convert_probabirity_to_self_information(
        convert_freq_to_probabirity(counter),
    )


def convert_freq_to_probabirity(freq: Dict[str, int]) -> Dict[str, float]:
    denominator = defaultdict(int)
    for k, v in counter.items():
        denominator[k[:2]] += v
    rst = {}
    for k, v in counter.items():
        rst[k] = v / denominator[k[:2]]
    return rst


def convert_probabirity_to_self_information(
    probabirity: Dict[str, float]
) -> Dict[str, float]:
    return {k: -1 * math.log2(v) for k, v in probabirity.items()}


# def calculate_weights(sequence: SeqRecord, n_split: int) -> Counter:
#     return Counter(gbk_utils.window_search(sequence, n_split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate weights of each N sequence based on self information.",
    )
    parser.add_argument(
        "gbks",
        nargs="*",
        help="Genbank files that count frquencies. It use json, this will none",
    )
    parser.add_argument(
        "--n_split",
        default=3,
        type=int,
        help="How many characters to split each sequence",
    )
    parser.add_argument(
        "--json",
        nargs=1,
        help="Use json that written information",
    )
    parser.add_argument(
        "--gbk_root",
        help="If use json, must be specify this",
    )
    parser.add_argument(
        "--split_taxon",
        action="store_true",
        help="Count based on taxon",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_false",
        help="Show progress bar",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Outout file(json)",
    )
    args = parser.parse_args()

    rst = {}
    if len(args.gbks) > 0:
        counter = Counter()
        for filename in tqdm(args.gbks, disable=args.verbose):
            for record in SeqIO.parse(filename, "genbank"):
                counter += Counter(
                    gbk_utils.window_search(
                        re.sub("[^ATGCatgc]", "", str(record.seq)),
                        args.n_split,
                    )
                )
        rst = convert_freq_to_self_information(counter)
    else:
        for filename in args.json:
            with open(filename) as f:
                information = json.load(f)
        if args.split_taxon:
            # make split like {"Mammala": {"AAA": 1.0, ...}}
            counter = defaultdict(Counter)
            for accession, taxon in tqdm(information.items(), disable=args.verbose):
                p = Path(args.gbk_root) / f"{accession}.gbk"
                for record in SeqIO.parse(p, "genbank"):
                    counter[taxon] += Counter(
                        gbk_utils.window_search(
                            re.sub("[^ATGCatgc]", "", str(record.seq)),
                            args.n_split,
                        )
                    )
            rst = {k: convert_freq_to_self_information(v) for k, v in counter.items()}
        else:
            counter = Counter()
            for accession in tqdm(information.keys(), disable=args.verbose):
                p = Path(args.gbk_root) / f"{accession}.gbk"
                for record in SeqIO.parse(p, "genbank"):
                    counter += Counter(
                        gbk_utils.window_search(
                            re.sub("[^ATGCatgc]", "", str(record.seq)),
                            args.n_split,
                        )
                    )
            rst = convert_freq_to_self_information(counter)

    with open(args.output, "w") as f:
        json.dump(rst, f)
