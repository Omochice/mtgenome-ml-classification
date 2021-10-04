import yaml
import argparse
from mizlab_tools.calculate_coordinates import calc_coord
from mizlab_tools.calculate_weights import calc_weights
from Bio import SeqIO, SeqRecord
import json
from collections import Counter
from tqdm import tqdm

from typing import Dict, Iterable
from glob import glob

from pathlib import Path
import pandas as pd
from typing import Iterable, List, Iterator


def filter_usegbk(gbkfiles: Iterable[str],
                  focus_rank: str,
                  information: Dict) -> Iterator[SeqRecord.SeqRecord]:
    # filter use gbk
    candidates = []
    for gbkfile in gbkfiles:
        for record in SeqIO.parse(gbkfile, "genbank"):
            info = informations[record.name]
            if (info["is_complete"] and not info["is_mongrel"]
                    and not info["is_shotgun"] and not info["is_chromosome"]):
                candidates.append(record)
            else:
                del information[record.name]

    invalids = {"null", "nan", None}
    n_classes = Counter([v["class"] for v in informations.values()])
    for record in candidates:
        if (n_classes[information[record.name][focus_rank]] >= 5
                and information[record.name][focus_rank] not in invalids):
            # yield record
            yield record, information[record.name][focus_rank]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("gbkfiles", nargs="+")
    parser.add_argument("--taxon", "-t", required=True)
    args = parser.parse_args()

    # project_dir = Path(__file__).parents[2]
    # print(project_dir)

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    df = pd.read_table(args.taxon, index_col=0).dropna(
        subset=[config["focus_rank"]], how="any")
    # n_classes = Counter(taxon.values())
    informations = {}
    for row in df.itertuples():
        informations[row[1]] = {}
        for key, value in zip(df.columns, row[1:]):
            informations[row[1]][key] = value

    del df  # this DataFrame is big object

    # gen weights
    # use_records = tuple(filter_usegbk(
    #     args.gbkfiles, config["focus_rank"], informations))
    use_records = []
    acc2class = {}
    for record, class_name in filter_usegbk(args.gbkfiles,
                                            config["focus_rank"],
                                            informations):
        use_records.append(record)
        acc2class[record.name] = class_name

    with (Path(config["data_dst"]) / "json" / "acc2class.json").open("w") as f:
        json.dump(acc2class, f)

    weights = calc_weights(use_records, "ATGC")
    with (Path(config["data_dst"]) / "json" / "weights.json").open("w") as f:
        json.dump(weights, f)

    # gen coordinates

    data_dst = Path(config["data_dst"]) / "coordinates"
    data_dst.mkdir(parents=True, exist_ok=True)
    for record in tqdm(use_records):
        coord = calc_coord(record, mapping=config["mapping"], weight=weights)
        acc = record.name
        with (data_dst / f"{acc}.json").open("w") as f:
            json.dump(coord, f)
