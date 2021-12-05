import argparse
import datetime
import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import models
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import visualize
import yaml
from focal_loss import categorical_focal_loss
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping,
                                        History, ModelCheckpoint)
# from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Dropout, Input
# from tensorflow.keras.utils import model_to_dot, plot_model
from tqdm import tqdm

Openable = Union[str, bytes, "os.PathLike[Any]"]


def to_categorical(labels: np.ndarray) -> np.ndarray:
    """Convert labels into `one-hot` style
    It will be acsending.
    ex) input  ['A', 'A', 'B', 'C', 'C', 'C']
        output [[0,1,0],[0,1,0],[1,0,0],[0,0,1],[0,0,1],[0,0,1]]

    Args:
        labels (np.ndarray): one-hotに変換するnumpy配列

    Returns:
        np.ndarray:
    """

    def _label_to_number(labels: np.ndarray, uniq: np.ndarray) -> np.ndarray:
        """ユニークな配列を基準に数値の配列を返す

        Args:
            labels (np.ndarray): 数値に変換するnumpy配列
            uniq (np.ndarray): 基準となるユニークな配列

        Returns:
            np.ndarray:
        """
        numbers = np.array([np.where(ll == uniq) for ll in labels])
        return numbers

    uniq = get_sorted_class(labels)
    numerical_labels = _label_to_number(labels, uniq)
    if not numerical_labels.ndim == 1:
        numerical_labels = numerical_labels.flatten()
    one_hot = np.identity(np.max(numerical_labels) + 1)[numerical_labels]
    return one_hot


def get_sorted_class(classes: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Get acsending ordered unique array.

    Args:
        classes (np.ndarray): 対象のnumpy配列

    Returns:
        np.ndarray:
    """
    counter = Counter(classes)
    return np.array(
        [arr[0] for arr in sorted(counter.items(), key=lambda x: x[1], reverse=reverse)]
    )


def load_images(paths: Iterable[Openable], as_monochrome: bool = True) -> np.ndarray:
    """Load images with trainable form.

    Args:
        paths (Iterable[PathLike]): Iterable object of Openable
        as_monochrome (bool): load image as monochrome.

    Returns:
        np.ndarray: The array of images
    """
    if as_monochrome:
        return np.array(
            [
                (np.array(Image.open(p).convert("L")).astype("float32")) / 255
                for p in paths
            ]
        )
    else:
        return np.array(
            [
                (np.array(Image.open(p).convert("RGB")).astype("float32")) / 255
                for p in paths
            ]
        )


class F1Callback(Callback):
    def __init__(self, print_on_epoch_end: bool = False):
        self.f1s = []
        self._print_pn_epoch_end = print_on_epoch_end

    def on_epoch_end(self, epoch, logs):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        if self._print_pn_epoch_end:
            print(f"f1_val = {f1}")
        self.f1s.append(f1)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))


def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))


def calculate_class_weights(labels: np.ndarray, option: str = None) -> Dict[int, int]:
    """Calculate class weights with some metrics.

    Args:
        labels (np.ndarray): labels like [1, 0, 2, 3, ...]
        option (str): option

    Returns:
        Dict[int, int]:
    """
    weights = compute_class_weight("balanced", np.unique(labels), labels)
    if option is None:
        return {i: weight for i, weight in enumerate(weights)}
    elif option == "log":
        weights = 1 / (len(weights) * weights)
        return {i: weight for i, weight in enumerate(-np.log(weights))}
    elif option == "normalized_log":
        weights = 1 / (len(weights) * weights)
        weights = -np.log(weights)
        x_max, x_min = max(weights), min(weights)
        return {
            i: (9 * (weight - x_min) / (x_max - x_min)) + 1
            for i, weight in enumerate(weights)
        }
    elif option == "ceil":
        return {i: max(1, weight) for i, weight in enumerate(weights)}
    elif option == "one":
        return {i: 1.0 for i in range(len(weights))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute machine-learing with graphed image of mitochondrion genome.",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="the path to config.yml",
        required=True,
    )
    parser.add_argument(
        "--acc2class",
        help="the path to acc2class.json",
        required=True,
    )
    parser.add_argument(
        "--use_model_name",
        help="use model name",
        default="mobilenetV2",
    )
    parser.add_argument(
        "--use_xy_only",
        help="use only xy dim images",
        action="store_true",
    )
    parser.add_argument(
        "--as_color",
        help="When use_xy_only is True, load image as color one",
        action="store_true",
    )
    parser.add_argument(
        "--parallel",
        help="Train each dimention image in different cnn",
        action="store_true",
    )
    parser.add_argument(
        "--description",
        "-d",
        help="description that output to json",
        type=str,
        default="",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dst = Path(config["data_dst"])

    with open(args.acc2class) as f:
        acc2class = json.load(f)

    img_dst = data_dst / "img"
    raw_labels = []
    tmp_paths = []
    for acc, cl in acc2class.items():
        raw_labels.append(cl)
        tmp_paths.append(str(img_dst / acc))
    raw_labels = np.array(raw_labels)

    # convert label to numpy array
    labels = to_categorical(raw_labels)
    chanel = 3

    if args.use_xy_only:
        if args.as_color:
            stacked_images = load_images(
                map(lambda x: x + "_0.png", tmp_paths), as_monochrome=False
            )
        else:
            stacked_images = load_images(
                map(lambda x: x + "_0.png", tmp_paths), as_monochrome=True
            )
            chanel = 1
    else:
        xy_images = load_images(map(lambda x: x + "_0.png", tmp_paths))
        yz_images = load_images(map(lambda x: x + "_1.png", tmp_paths))
        zx_images = load_images(map(lambda x: x + "_2.png", tmp_paths))
        stacked_images = np.stack((xy_images, yz_images, zx_images), axis=3)

    sorted_unique_label = get_sorted_class(raw_labels)

    # other settings
    n_class = len(sorted_unique_label)
    weights = calculate_class_weights(np.argmax(labels, axis=1), option=None)
    results = {"results": []}
    # If training failled(when accuracy/f1score not move), retry training.
    threshold = {"acc": 0.3, "f1": 0.1}

    skf = StratifiedKFold(config["use_limit"])
    study_log = {"acc": [], "f1": [], "macrof1": []}
    now = datetime.datetime.now().strftime("%m%d-%H%M")
    log_dst = data_dst.parent / "log" / now
    log_dst.mkdir(parents=True, exist_ok=True)
    image_shape = (config["graph_pix"], config["graph_pix"], chanel)

    # Dump local setiings
    with (log_dst / "conf.json").open("w") as f:
        json.dump(
            dict(args.__dict__, **{"threshold": "threshold"}),
            f,
            ensure_ascii=False,
        )

    for i, (train_index, test_index) in enumerate(
        skf.split(stacked_images, raw_labels), 1
    ):
        trial_dst = log_dst / str(i)
        trial_dst.mkdir(exist_ok=True, parents=True)
        if args.parallel:
            # split to train and test
            xy_train, xy_test = xy_images[train_index], xy_images[test_index]
            yz_train, yz_test = yz_images[train_index], yz_images[test_index]
            zx_train, zx_test = zx_images[train_index], zx_images[test_index]
            image_train = [xy_train, yz_train, zx_train]
            image_test = [xy_test, yz_test, zx_test]
        else:
            image_train, image_test = (
                stacked_images[train_index],
                stacked_images[test_index],
            )
        label_train, label_test = labels[train_index], labels[test_index]
        raw_label_test = raw_labels[test_index]

        # re-train if result score is too bad
        for _ in range(config["n_trial_in_ml"]):
            model = models.construct_model(
                input_shape=image_shape,
                n_class=n_class,
                parallel=args.parallel,
                use_model_name=args.use_model_name,
                n_dense_dropout=2,
                use_imagenet=True,
                show_model=False,
                save_model=True,
                filename="model",
                dst_dir=log_dst,
                extent="svg",
            )

            # callbacks
            csv_logger = CSVLogger(trial_dst / "logger.csv")
            early_stopping = EarlyStopping(monitor="val_loss", patience=100, mode="min")
            f1cb = F1Callback()
            history = History()
            checkpoint = ModelCheckpoint(
                trial_dst / "weights_{epoch:03d}.hdf5",
                monitor="val_loss",
                # save_best_only=True,
                save_weights_only=True,
            )

            # compile model
            model.compile(
                loss=[
                    categorical_focal_loss(
                        alpha=[[0.25 for _ in range(n_class)]], gamma=2
                    )
                ],
                metrics=[
                    "accuracy",
                    true_positives,
                    possible_positives,
                    predicted_positives,
                ],
                # metrics=["accuracy"],
                optimizer="Adam",
            )

            # fit
            history = model.fit(
                image_train,
                label_train,
                validation_data=(image_test, label_test),
                epochs=1000,
                batch_size=16,
                callbacks=[csv_logger, checkpoint, early_stopping, f1cb],
                class_weight=weights,
            )

            # evaluate scores at best scored model
            best_f1_epoch = f1cb.f1s.index(max(f1cb.f1s)) + 1  # it is 1 origin
            best_model_name = f"weights_{best_f1_epoch:03d}.hdf5"

            # remove weight file that not have best f1 score.
            for weight_file_name in trial_dst.glob("*.hdf5"):
                if weight_file_name.name != best_model_name:
                    os.remove(weight_file_name)

            model.load_weights(trial_dst / best_model_name)
            loss, acc, *_ = model.evaluate(image_test, label_test, verbose=1)
            pred_labels = np.argmax(model.predict(image_test), axis=1)

            # f1 = f1_score(np.argmax(label_test, axis=1), pred_labels, average="micro",
            #               zero_division=0)
            macrof1 = f1_score(
                np.argmax(label_test, axis=1),
                pred_labels,
                average="macro",
                zero_division=0,
            )

            if macrof1 > threshold["f1"]:
                # if complete training 3times or score good point,
                # break this for loop.
                break

        results["results"].append(
            {
                f"trial_{i}": {
                    "Accuracy": float(acc),
                    "F1 score": macrof1,
                }
            }
        )
        study_log["acc"].append(float(acc))
        # study_log["f1"].append(f1)
        study_log["f1"].append(macrof1)
        # study_log["macrof1"].append(macrof1)

        # visualizing
        visualize.visualize_history(history.history, "study_log", trial_dst)
        visualize.plot_cmx(
            np.argmax(label_test, axis=1),
            pred_labels,
            sorted_unique_label,
            title="cmx",
            dst=trial_dst,
        )
        with (trial_dst / "trainlog.json").open("w") as f:
            json.dump(history.history, f, indent=2)

        with (trial_dst / "report.txt").open("w") as f:
            print(
                classification_report(
                    np.argmax(label_test, axis=1),
                    pred_labels,
                    target_names=sorted_unique_label,
                    zero_division=0,
                ),
                file=f,
            )

    with (log_dst / "weight.json").open("w") as f:
        json.dump(
            {str(k): weights[i] for i, k in enumerate(get_sorted_class(raw_labels))},
            f,
            indent=2,
        )

    results["average"] = {
        "Accuracy": sum(study_log["acc"]) / len(study_log["acc"]),
        "F1 score": sum(study_log["f1"]) / len(study_log["f1"]),
        # "MacroF1": sum(study_log["macrof1"]) / len(study_log["macrof1"])
    }

    tmp_log = {k: [] for k in study_log.keys()}
    for i in range(config["use_limit"]):
        if all([study_log[k][i] > threshold[k] for k in ("acc", "f1")]):
            for k in ("acc", "f1"):
                tmp_log[k].append(study_log[k][i])

    results["外れ値を除外"] = {
        "Accuracy": sum(tmp_log["acc"]) / len(tmp_log["acc"]),
        "F1 score": sum(tmp_log["f1"]) / len(tmp_log["f1"]),
        # "MacroF1": sum(study_log["macrof1"]) / len(study_log["macrof1"])
    }

    with (log_dst / "results.json").open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trials = ["trial_" + str(i + 1) for i in range(config["use_limit"])]
    visualize.show_cmx_gif(
        results=[results["results"][i][t] for i, t in enumerate(trials)],
        cmxs=[log_dst / str(i + 1) / "cmx.png" for i in range(config["use_limit"])],
        dst=log_dst,
    )
