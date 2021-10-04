from pathlib import Path
import argparse

import yaml


def init_settig(path: Path) -> None:
    if path.exists():
        print(f"{str(path)} exists already.")
        return
    project_dir = path.parent
    priority = [
        "NCBI Taxonomy", "NCBI", "GBIF Backbone Taxonomy", "Catalogue of Life",
        "Open Tree of Life Reference Taxonomy",
        "The Interim Register of Marine and Nonmarine Genera"
    ]
    template = {
        "email": "<FILL IN>",
        "source_csv": "<FILL IN>",
        "data_dst": str(project_dir / "data"),
        "graph_pix": 192,
        "priority": priority,
        "focus_rank": "class",
        "use_limit": 5,
        "log_dst": str(project_dir / "log"),
        "mapping": {
            "A": [1, 1, 1],
            "T": [-1, 1, -1],
            "G": [-1, -1, 1],
            "C": [1, -1, -1],
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(template, f)
    print(f"GENERATING SETTING FILE. -> {str(path)}")
    print("generating is done. Please rewrite if you need.")

    (project_dir / "data").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    default_dst = Path(__file__).resolve().parents[1] / "setting.yml"
    parser = argparse.ArgumentParser(description="Generate config.yml.")
    parser.add_argument("--destination",
                        "-d",
                        type=str,
                        help=f"If specify this, generate config to there. default={str(default_dst)}",)
    args = parser.parse_args()

    if args.destination:
        init_settig(default_dst)
    else:
        init_settig(Path(args.destination))
