import argparse
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from Bio import SeqIO
from mizlab_tools import gbk_utils
from tqdm import tqdm

flatten = itertools.chain.from_iterable


def get_priority(taxon: dict, priority: List[str], rank: str) -> Optional[str]:
    for db_name in priority:
        taxon_name = taxon.get(db_name, {}).get(rank, None)
        if taxon_name is not None:
            return taxon_name
    return None


@dataclass
class Invalid:
    expr: Callable
    description: str
    accessions: List[str] = field(default_factory=list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "taxon",
        help="json file that has taxonomy information",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="config.yml that has `priority`",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show number of species that is filterd",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="filter species that len(taxon that include) < 5",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output file path.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        priority = config["priority"]
        gbk_root = Path(config["data_dst"]) / "gbk"
        focus_rank = config["focus_rank"]

    with open(args.taxon) as f:
        taxon = json.load(f)

    # NOTE: expr that definition on here expect to be gave argument SeqIO.SeqRecord.
    invalids: List[Invalid] = [
        Invalid(
            lambda x: "shotgun" in gbk_utils.get_definition(x), "It is shotgun sequence"
        ),
        Invalid(
            lambda x: "complete" not in gbk_utils.get_definition(x),
            "It is not complete sequence",
        ),
        Invalid(
            lambda x: "chromosome" in gbk_utils.get_definition(x),
            "It is chromosome sequence",
        ),
        Invalid(
            lambda x: "assembly" in gbk_utils.get_definition(x),
            "It is assembly sequence",
        ),
        Invalid(
            lambda x: "partial" in gbk_utils.get_definition(x), "It is partial sequence"
        ),
        Invalid(lambda x: gbk_utils.has_contig(x), "It has contig"),
        Invalid(lambda x: " x " in gbk_utils.get_creature_name(x), "It is mongrel"),
    ]

    acc2taxon = {}
    for accession, values in tqdm(taxon.items()):
        # NOTE: None is invalid taxon name.
        acc2taxon[accession] = get_priority(values["taxon"], priority, focus_rank)

        with (gbk_root / f"{accession}.gbk").open() as f:
            for record in SeqIO.parse(f, "genbank"):
                for inv in invalids:
                    if inv.expr(record):
                        inv.accessions.append(accession)

    if args.filter:
        invalids.append(
            Invalid(
                lambda: None,
                description="It have no class",
                accessions=[acc for acc, cl in acc2taxon.items() if cl is None],
            )
        )
        for k in set(flatten([inv.accessions for inv in invalids])):
            del acc2taxon[k]
        # n < 5 must do on hook_after
        n_class = Counter(acc2taxon.values())
        n_five = [acc for acc, cl in acc2taxon.items() if n_class[cl] < 5]
        for k in n_five:
            del acc2taxon[k]
        invalids.append(
            Invalid(
                lambda: None,
                description="Number of its class is < 5",
                accessions=n_five,
            )
        )

        if args.verbose:
            for inv in invalids:
                print(f"{inv.description}: {len(inv.accessions)}")

    with open(args.out, "w") as f:
        json.dump(acc2taxon, f)
