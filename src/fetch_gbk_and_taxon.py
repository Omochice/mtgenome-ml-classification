import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, Union, Tuple, List, Optional
from collections import Counter, OrderedDict

import mizlab_tools.fetch_gbk
import mizlab_tools.fetch_taxon
import pandas as pd
import yaml
from Bio import SeqIO, SeqRecord


def fetch_gbk(accs: Iterable[str], dst_root: Path, email: str):
    dst = dst_root / "gbk"
    dst.mkdir(parents=True, exist_ok=True)

    for gbk in mizlab_tools.fetch_gbk.fetch(accs, email=email):
        with open(dst / f"{gbk.name}.gbk", "w") as f:
            SeqIO.write(gbk, f, "genbank")


def fetch_contigs(search_root: Path, email=str):
    records_has_contig = []
    contigs = []
    for gbk in glob(str(search_root / "*.gbk")):
        for record in SeqIO.parse(gbk, "genbank"):
            if mizlab_tools.gbk_utils.has_contig(record):
                records_has_contig.append(record)
                contigs.append(
                    mizlab_tools.gbk_utils.parse_contig(record.annotations["contig"]))

    contig_records = mizlab_tools.fetch_gbk.fetch([d["accession"] for d in contigs],
                                                  email=email)

    for original, contig_record, contig_info in zip(records_has_contig, contig_records,
                                                    contigs):
        seq = contig_record.seq[contig_info["start"]:contig_info["end"]]
        if contig_info["is_complement"]:
            seq = seq.complement()
        original.seq = seq
        del original.annotations["contig"]
        print(f"{original.name} has contig")
        with open(search_root / f"{original.name}.gbk", "w") as f:
            SeqIO.write(original, f, "genbank")


def fetch_taxon(gbkfiles: Iterable, email: str) -> Dict:
    records = []
    for gbk in gbkfiles:
        for record in SeqIO.parse(gbk, "genbank"):
            records.append(record)
    return mizlab_tools.fetch_taxon.fetch_taxon(records, email=email)


def parse_csv(src) -> Iterator[str]:
    df = pd.read_csv(src, header=0)
    return map(lambda x: x.split("/")[0].split(":")[-1], df["Replicons"])


def get_taxon(priority, taxon: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    for p in priority:
        if p in taxon.keys() and "class" in taxon[p].keys():
            return taxon[p]["class"], p
    return None, None


def make_csv(records: Iterable[SeqRecord.SeqRecord], taxon: dict, priority: Union[List[str], Tuple[str]]) -> pd.DataFrame:
    informations = []
    for record in records:
        acc = record.name
        cl_name, source = get_taxon(priority, taxon[acc]["taxon"])
        seq_counter = Counter(str(record.seq))
        informations.append({
            "accession": acc,
            "binomial_name": mizlab_tools.gbk_utils.get_creature_name(record),
            "class": cl_name,
            "source_db": source,
            "is_mongrel": mizlab_tools.gbk_utils.is_mongrel(record),
            "is_complete": "complete" in mizlab_tools.gbk_utils.get_definition(record),
            "is_shotgun": "shotgun" in mizlab_tools.gbk_utils.get_definition(record),
            "is_chromosome": "chromosome" in mizlab_tools.gbk_utils.get_definition(record),
            "seq_len": len(record.seq),
            "rate_a": seq_counter["A"],
            "rate_t": seq_counter["T"],
            "rate_g": seq_counter["G"],
            "rate_c": seq_counter["C"],
            "rate_other": len(record.seq) - sum([seq_counter[base] for base in "ATGC"]),
        })
    return pd.DataFrame(informations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    csv = config["source_csv"]
    data_dst = Path(config["data_dst"])
    fetch_gbk(csv, Path(config["data_dst"]), config["email"])
    fetch_contigs(Path(config["data_dst"])/"gbk", config["email"])
    with open(data_dst / "json" / "taxon.json", "w") as f:
        json.dump(
            fetch_taxon(glob(str(data_dst / "gbk" / "*.gbk")), email=config["email"]),
            f)
    records = []
    for r in glob(str(data_dst/"gbk"/"*.gbk")):
        for record in SeqIO.parse(r, "genbank"):
            records.append(record)
    with open(data_dst/"json"/"taxon.json") as f:
        df = make_csv(records, json.load(f), config["priority"])
        df.to_csv(data_dst/"csv"/"informations.csv", sep="\t")

    with open(data_dst/"json"/"n_classes.json", "w") as f:
        frozen = OrderedDict(sorted(Counter(df["class"]).items(), key=lambda x: -x[1]))
        json.dump(frozen, f, indent=2)
