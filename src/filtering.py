import argparse
from tqdm import tqdm
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from Bio import SeqIO
from mizlab_tools import gbk_utils


def get_priority(taxon: dict, priority: List[str], rank: str) -> Optional[str]:
    for db_name in priority:
        taxon_name = taxon.get(db_name, {}).get(rank, None)
        if taxon_name is not None:
            return taxon_name
    return None


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
        "--show_filtered",
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

    selected = {}
    n_mongrel = 0
    accs_has_contig = []
    accs_has_shotgun = []
    accs_has_chromosome = []
    for accession, values in tqdm(taxon.items()):
        if " x " in values["binomial_name"]:
            n_mongrel += 1
        taxon_name = get_priority(values["taxon"], priority, focus_rank)
        if taxon_name is not None:
            selected[accession] = taxon_name
        with (gbk_root / f"{accession}.gbk").open() as f:
            for record in SeqIO.parse(f, "genbank"):
                definition = gbk_utils.get_definition(record)
                if gbk_utils.has_contig(record):
                    accs_has_contig.append(accession)
                if "shotgun" in definition:
                    accs_has_shotgun.append(accession)
                if "chromosome" in definition:
                    accs_has_chromosome.append(accession)

    c = Counter(selected.values())
    n_five = [k for k, v in selected.items() if c[v] < 5]

    if args.filter:
        for k in set(
            n_five + accs_has_contig + accs_has_shotgun + accs_has_chromosome
        ):
            del selected[k]

    if args.show_filterd:
        print(f"mongrel: {n_mongrel}")
        print(f"No class: {len(n_five)}")
        print(f"n<5: {len([None for v in c.values() if v < 5])}")
        print(f"contig: {len(accs_has_contig)}")
        print(f"shotgun: {len(accs_has_shotgun)}")
        print(f"chromosome: {len(accs_has_chromosome)}")

    with open(args.out, "w") as f:
        json.dump(selected, f)
