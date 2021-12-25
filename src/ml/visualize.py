import argparse
import itertools
import os
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

sns.set()
matplotlib.use("Agg")


def __load_japanize_matplotlib() -> None:
    """Lazy loading to load japanize matplotlib after matplotlib

    Returns:
        None:
    """
    import japanize_matplotlib


__load_japanize_matplotlib()


def visualize_history(
    history: dict,
    title: str = "",
    dst: PathLike = "",
) -> None:
    """Visialize training history into image.

    Args:
        history (dict): history.history
        title (str): Thet title string for image.
        dst (PathLike): The path to save image.

    Returns:
        None:
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    axL.plot(
        history.get("acc", history.get("accuracy", None)),
        "o-",
        label="Train accuracy",
    )
    axL.plot(
        history.get("val_acc", history.get("val_accuracy", None)),
        "o-",
        label="Validation accuracy",
    )
    if "f1score" in history:
        axL.plot(history["f1score"], "*-", label="F1 score")
        axL.set_title("Accuracy and F1 score")
        axL.set_ylabel("value of score")
    else:
        axL.set_title("Accuracy")
        axL.set_ylabel("Accuracy")
    axL.set_xlabel("Epoch")
    axL.set_ylim(0, 1)
    axL.grid(True)
    # axL.legend(bbox_to_anchor=(0, 0), loc="lower left", borderaxespad=0)
    axL.legend(bbox_to_anchor=(1, -0.1), loc="upper right", borderaxespad=0)
    # axL.legend(loc="best")

    axR.plot(history["loss"], "o-", label="Train loss")
    axR.plot(history["val_loss"], "o-", label="Validation loss")
    axR.set_title("Loss")
    axR.set_xlabel("Epoch")
    axR.set_ylabel("Loss")
    axR.grid(True)
    # axR.legend(bbox_to_anchor=(0, 0), loc="lower left", borderaxespad=0)
    axR.legend(bbox_to_anchor=(1, -0.1), loc="upper right", borderaxespad=0)
    # axR.legend(loc="best")

    # TODO
    # 3つ目のグラフの作成

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if title == "":
        title = "model_history"
    fig.savefig(os.path.join(dst, f"{title}.png"), bbox_inches="tight", pad_inches=0.05)
    fig.clf()


def plot_cmx(
    y_true: Union[np.ndarray, List[Union[int, float]]],
    y_pred: Union[np.ndarray, List[Union[int, float]]],
    labels: List[str],
    title: str,
    dst: Union[str, Path],
    emphasize_diagonal_elements: bool = False,
):
    """Generate figure that shows confusion-matrics of machine learning result

    Args:
        y_true (Union[np.ndarray, List[Union[int, float]]]): List of true labels(not one-hoted)
        y_pred (Union[np.ndarray, List[Union[int, float]]]): List of predicted labels(not one-hoted)
        labels (List[str]): Label names. this is used as tick labels
        title (str): Title string for figure
        dst (Union[str, Path]): destination path for save figure
        emphasize_diagonal_elements (bool): Whether emphasize diagonal elements
    """
    fig, ax = plt.subplots(figsize=(15, 13))
    cmx = confusion_matrix(y_true, y_pred)
    normalized_cmx = [row / np.sum(row) for row in cmx]

    if emphasize_diagonal_elements:
        # write value in cells
        # annot can not specify diagonal elements
        for y, x in itertools.product(range(len(cmx)), range(len(cmx))):
            if x == y:
                color = "red"  # it is diagonal elements
            elif normalized_cmx[y][x] > 0.5:
                color = "white"  # cell background is like black
            else:
                color = "black"  # cell background is like white
            ax.text(
                x + 0.5,
                y + 0.5,
                cmx[y][x],
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
            )
    else:
        sns.heatmap(
            normalized_cmx,
            xticklabels=labels,
            yticklabels=labels,
            annot=cmx,
            square=True,
            cmap="Blues",
            vmin=0,
            vmax=1.0,
            ax=ax,
            fmt="d",
        )

    ax.set_title("正解の綱と予測した綱の対応")
    ax.set_xlabel("予測した綱")
    ax.set_ylabel("正解の綱")
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.05)
    fig.clf()


def show_cmx_gif(
    results: List[Dict[str, float]],
    cmxs: Iterable[PathLike],
    dst: PathLike,
    row_idxs: List[str] = ["trial_" + str(i) for i in range(1, 6)],
    show_execlude_outlier: bool = True,
    threshold: Dict[str, float] = {"Accuracy": 0.3, "F1 score": 0.1},
    keys: Iterable[str] = ("Accuracy", "F1 score"),
):
    """Generate results of some K-fold and their table

    Args:
        results (List[Dict[str, float]]): Results to generate table
        cmxs (Iterable[PathLike]): The paths to confusion matricses
        dst (PathLike): The path to save figure
        row_idxs (List[str]): row_idxs
        show_execlude_outlier (bool): show_execlude_outlier
        threshold (Dict[str, float]): threshold
        keys (Iterable[str]): keys
    """
    assert results[0].keys() == threshold.keys()
    # make 2d list
    data = [[d[k] for k in keys] for d in results]
    dst = Path(dst)

    ave = []
    for i in range(len(keys)):
        ave.append(sum([d[i] for d in data]) / len(data))
    data.append(ave)
    row_idxs.append("Average")

    if show_execlude_outlier:
        excluded_data = []
        for row in data[:-1]:
            if all([row[i] > threshold[k] for i, k in enumerate(keys)]):
                excluded_data.append(row)
        if len(excluded_data) < len(data) - 1:
            ex_ave = []
            for i in range(len(keys)):
                ex_ave.append(sum([d[i] for d in excluded_data]) / len(excluded_data))
            data.append(ex_ave)
            row_idxs.append("外れ値を除く")

    # convert float -> str
    converted = [[f"{v:.3f}" for v in row] for row in data]

    # concate table and cmxs
    images = []
    for i, img_path in enumerate(cmxs, 1):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        ax1.axis("off")
        tb = ax1.table(
            cellText=converted,
            colLabels=keys,
            rowLabels=row_idxs,
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        for j in range(len(keys)):
            tb[0, j].set_facecolor("#363636")
            tb[0, j].set_text_props(color="w")
            tb[i, j].set_facecolor("tomato")

        ax2.axis("off")
        img = plt.imread(img_path)
        ax2.imshow(img)
        filedst = dst / f".tmp_{i}.png"
        fig.savefig(filedst, bbox_inches="tight", pad_inches=0.05)
        images.append(Image.open(filedst))
        os.remove(filedst)

    # make gif animation
    images[0].save(
        dst / "log.gif",
        format="gif",
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0,
    )


# def visualize_all_cmxs(
#     rst: list,
#     cmx_pathes: Iterable[PathLike],
#     ave: dict,
#     dst: PathLike,
#     show_execlude_outlier: bool = False,
# ) -> None:
#     """visualize_all_cmxs.
#
#     Args:
#         rst (list): rst
#         cmx_pathes (Iterable[PathLike]): cmx_pathes
#         ave (dict): ave
#         dst (PathLike): dst
#         show_execlude_outlier (bool): show_execlude_outlier
#
#     Returns:
#         None:
#     """
#     images = []
#     dst = Path(dst)
#     # make table
#     d = {}
#     average_widthout_outlier = {}
#     for row in rst:
#         for k, v in row.items():
#             d[k] = {kk: f"{vv:.3f}" for kk, vv in v.items()}
#
#     d["Average"] = {k: f"{v:.3f}" for k, v in ave.items()}
#
#     if show_execlude_outlier:
#         for k in average_widthout_outlier.keys():
#             average_widthout_outlier[k] = sum(average_widthout_outlier[k]) / len(
#                 average_widthout_outlier[k]
#             )
#         d["外れ値を除く"] = {k: f"{v:.3f}" for k, v in average_widthout_outlier.items()}
#     df = pd.DataFrame.from_dict(d, orient="index")
#
#     # concate table and cmxs
#     for i, img_path in enumerate(cmx_pathes, 1):
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
#         ax1.axis("off")
#         tb = ax1.table(
#             cellText=df.values,
#             colLabels=df.columns,
#             rowLabels=df.index,
#             loc="center",
#             bbox=[0, 0, 1, 1],
#         )
#         for j in range(len(df.columns)):
#             tb[0, j].set_facecolor("#363636")
#             tb[0, j].set_text_props(color="w")
#             tb[i, j].set_facecolor("tomato")
#
#         ax2.axis("off")
#         img = plt.imread(img_path)
#         ax2.imshow(img)
#         filedst = dst / f".tmp_{i}.png"
#         fig.savefig(filedst, bbox_inches="tight", pad_inches=0.05)
#         images.append(Image.open(filedst))
#         os.remove(filedst)
#
#     # make gif animation
#     images[0].save(
#         dst / "log.gif",
#         format="gif",
#         save_all=True,
#         append_images=images[1:],
#         duration=1000,
#         loop=0,
#     )


def _history(args: argparse.Namespace):
    """_history.

    Args:
        args (argparse.Namespace): args
    """
    with open(args.source) as f:
        d = json.load(f)

    dst = Path(args.dst).resolve()
    visualize_history(d, args.title, dst)


def _all_cmx(args: argparse.Namespace):
    """_all_cmx.

    Args:
        args (argparse.Namespace): args
    """
    with open(args.source) as f:
        d = json.load(f)
    row_labels = ["trial_" + str(i) for i in range(1, 6)]
    data = []
    for i, k in enumerate(row_labels):
        data.append(d["results"][i][k])
    show_cmx_gif(data, args.cmxs, args.dst)
    # rst = [
    #     {k: d[k]} for k
    # ]
    # rst = [
    #     {k:
    #      {
    #          use_key: v[use_key] for use_key in args.use_keys
    #      }
    #      for k, v in trial.items()}
    #     for trial in d["results"]]
    # ave = {
    #     k: sum()
    # }
    # visualize_all_cmxs(rst, args.cmxs, ave, args.dst)


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Show train log as graph")
    subparsers = parser.add_subparsers()

    parser_history = subparsers.add_parser("history", help="Show history as graph")
    parser_history.add_argument("source", help="The source history file.")
    parser_history.add_argument(
        "--dst", "-d", help="Destination of output.", default="./output.png"
    )
    parser_history.add_argument("-t", "--title", help="The title of graph.", default="")
    parser_history.set_defaults(handler=_history)
    # parser.add_argument("source", help="Source log file")
    # parser.add_argument("--dst", "-d", help="Destination of output file.",
    #                     default=".")
    # parser.add_argument("--title", "-t", help="Graph title", default="title")
    # _history(parser.parse_args())

    parser_cmx = subparsers.add_parser(
        "all_cmx", help="Show joined result of each trial"
    )
    parser_cmx.add_argument(
        "source",
        help="The source file\nExpected {'results': List[Dict[str, int | float]]}",
    )
    parser_cmx.add_argument(
        "--dst", "-d", help="Destination of output", default="./output.gif"
    )
    # parser_cmx.add_argument("--use_keys", nargs="+", required=True,
    #                         help="Use key. it show as row titles")
    parser_cmx.add_argument(
        "--cmxs", nargs="+", required=True, help="The paths of use cmx image"
    )
    parser_cmx.set_defaults(handler=_all_cmx)

    # parser.add_argument("source", help="Source log file")
    # parser.add_argument("--dst", "-d", help="Destination of output file.",
    #                     default=".")
    # parser.add_argument("--cmxs", nargs="+")
    # args = parser.parse_args()
    # _all_cmx(args)

    #
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
