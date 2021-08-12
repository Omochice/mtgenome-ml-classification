import argparse
from tqdm import tqdm
import json
import yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# dont format
import matplotlib.pyplot as plt


def format_graph_img(ax: matplotlib.axes._axes.Axes, xmin: float, xmax: float,
                     ymin: float, ymax: float) -> None:
    """縦横比を揃えたりする

    Args:
        fig (matplotlib.figure.Figure): グラフ画像

    Returns:
        matplotlib.figure.Figure: [description]
    """
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generage images")
    parser.add_argument("coordinates", nargs="+")
    parser.add_argument("--config", "-c", help="the path to config.yml")
    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    pix_a_side = config["graph_pix"]
    dpi = 100
    figsize = (pix_a_side/dpi, pix_a_side/dpi)
    graph_dst = Path(config["data_dst"]) / "img"
    graph_dst.mkdir(parents=True, exist_ok=True)

    for coor_path in tqdm(map(lambda x: Path(x), args.coordinates)):
        with coor_path.open() as f:
            coords = json.load(f)
        accession = coor_path.stem
        for dim1, dim2 in tqdm(((0, 1), (1, 2), (2, 0)), leave=False, desc=accession):
            # may fast to use numpy like coords[:, :, 1] ...
            dims1 = [c[dim1] for c in coords]
            dims2 = [c[dim2] for c in coords]
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.plot(dims1, dims2, color="Black", lw=1)
            format_graph_img(ax, min(dims1), max(dims1), min(dims2), max(dims2))
            dst = graph_dst / f"{accession}_{dim1}.png"
            plt.savefig(dst)
            plt.close()
