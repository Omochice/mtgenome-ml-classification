from sklearn.utils.class_weight import compute_class_weight
import glob
from collections import Counter
import argparse
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from focal_loss import categorical_focal_loss
import tensorflow as tf
import visualize
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Concatenate
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small
from tensorflow.keras.utils import plot_model, model_to_dot
from tensorflow.keras.callbacks import (
    CSVLogger, History, ModelCheckpoint, EarlyStopping)
from typing import Tuple, Union, Dict, Any, Iterable
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score

import pandas as pd
from tqdm import tqdm
from collections import Counter
import yaml
import json

import os
import models

Openable = Union[str, bytes, "os.PathLike[Any]"]


# def construct_model(n_class: int, shape: Tuple[int, int]) -> tf.keras.Model:
#     """construct_model.
#
#     Args:
#         n_class (int): n_class
#         shape (Tuple[int, int]): shape
#
#     Returns:
#         tf.keras.Model:
#     """
#     input_ = Input(shape=(*shape, 3))
#     model = MobileNet(include_top=False,
#                       input_shape=(*shape, 3),
#                       weights="imagenet",
#                       pooling="max")(input_)
#     # weights="imagenet",
#     model = Dense(512, activation="relu")(model)
#     model = Dropout(0.3)(model)
#     model = Dense(512, activation="relu")(model)
#     model = Dropout(0.3)(model)
#     model = Dense(n_class, activation="softmax")(model)
#
#     return tf.keras.Model(inputs=input_, outputs=model,)
#
#
# def show_model(model: tf.keras.Model, dst: Openable, svg: bool = False) -> None:
#     """show_model.
#
#     Args:
#         model (tf.keras.Model): model
#         dst (Openable): dst
#
#     Returns:
#         None:
#     """
#     model.summary()
#     if svg:
#         model_to_dot(model, show_shapes=True).write_svg(dst)
#     else:
#         plot_model(model, to_file=dst, show_shapes=True)


def to_categorical(labels: np.ndarray) -> np.ndarray:
    """labelをone-hotなラベルに変更する
    labelの数値はラベルの出現数の昇順になる
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
    """get acsending ordered unique array.

    Args:
        classes (np.ndarray): 対象のnumpy配列

    Returns:
        np.ndarray:
    """
    counter = Counter(classes)
    return np.array([arr[0]
                     for arr in sorted(counter.items(),
                                       key=lambda x: x[1],
                                       reverse=reverse)])


def load_images(paths: Iterable[Openable], as_monochrome: bool = True) -> np.ndarray:
    """.

    Args:
        paths (Iterable[PathLike]): Iterable object of Openable
        as_monochrome (bool): load image as monochrome.

    Returns:
        np.ndarray: The array of images
    """
    if as_monochrome:
        return np.array([(np.array(Image.open(p).convert("L")).astype("float32")) / 255
                         for p in paths])
    else:
        return np.array([(np.array(Image.open(p).convert("RGB")).astype("float32")) / 255
                         for p in paths])


# class F1Callback(Callback):
#     """F1Callback.
#     """
#
#     def __init__(self, model, X_val, y_val):
#         """__init__.
#
#         Args:
#             model:
#             X_val:
#             y_val:
#         """
#         self.model = model
#         self.X_val = X_val
#         self.y_val = y_val
#         self.f1s = []
#
#     def on_epoch_end(self, epoch, logs):
#         """on_epoch_end.
#
#         Args:
#             epoch:
#             logs:
#         """
#         pred = self.model.predict(self.X_val)
#         f1_val = f1_score(self.y_val, np.round(pred), average="macro")
#         self.f1s.append(f1_val)
#         print("f1_val =", f1_val)

class F1Callback(Callback):
    def __init__(self):
        self.f1s = []

    def on_epoch_end(self, epoch, logs):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / \
            (logs["val_predicted_positives"] + eps)
        f1 = 2*precision * recall / (precision+recall+eps)
        print(f"f1_val = {f1}")
        self.f1s.append(f1)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))


def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))


def f1score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1-y_true) * (1-y_pred), "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1-y_pred), "float"), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def calcurate_class_weights(labels: np.ndarray, option: str = None) -> Dict[int, int]:
    """calcurate_class_weights.

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="the path to config.yml", required=True)
    parser.add_argument("--acc2class", help="the path to acc2class.json", required=True)
    parser.add_argument("--use_model_name", help="use model name",
                        default="mobilenetV2")
    parser.add_argument(
        "--use_xy_only", help="use only xy dim images", action="store_true")
    parser.add_argument(
        "--as_color", help="When use_xy_only is True, load image as color one", action="store_true")
    parser.add_argument(
        "--description", "-d",
        help="description that output to json",
        type=str,
        default=""
    )
    parser.add_argument(
        "--hogehoge",
        help="hoge",
        action="store_true"
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
                map(lambda x: x + "_0.png", tmp_paths),
                as_monochrome=False)
        elif args.hogehoge:
            xy = load_images(map(lambda x: x + "_0.png", tmp_paths),
                             as_monochrome=True)
            stacked_images = np.stack((xy, xy, xy), axis=3)
        else:
            stacked_images = load_images(
                map(lambda x: x + "_0.png", tmp_paths),
                as_monochrome=True)
            chanel = 1
    else:
        xy_images = load_images(map(lambda x: x + "_0.png", tmp_paths))
        yz_images = load_images(map(lambda x: x + "_1.png", tmp_paths))
        zx_images = load_images(map(lambda x: x + "_2.png", tmp_paths))
        stacked_images = np.stack((xy_images, yz_images, zx_images), axis=3)

    sorted_unique_label = get_sorted_class(raw_labels)
    # print(f"DEBUG {np.shape(stacked_images)}")
    # with open("debug.log", "w") as f:
    #     print(sorted_unique_label, file=f)
    #     print("-----", file=f)
    #     print(np.argmax(labels[:10], axis=1), file=f)
    #     print("-----", file=f)
    #     print(raw_labels[:10], file=f)

    # other settings
    n_class = len(sorted_unique_label)
    weights = calcurate_class_weights(np.argmax(labels, axis=1), option=None)
    results = {"results": []}
    threshold = {"acc": 0.3, "f1": 0.1}

    skf = StratifiedKFold(config["use_limit"])
    study_log = {"acc": [], "f1": [], "macrof1": []}
    now = datetime.datetime.now().strftime("%m%d-%H%M")
    log_dst = data_dst.parent / "log" / now
    log_dst.mkdir(parents=True, exist_ok=True)
    image_shape = (config["graph_pix"], config["graph_pix"], chanel)
    with (log_dst / "conf.json").open("w") as f:
        json.dump({"as_color": args.as_color,
                   "use_xy_only": args.use_xy_only,
                   "n_class": n_class,
                   "use_model_name": args.use_model_name,
                   "threshold": threshold,
                   "description": args.description
                   }, f, ensure_ascii=False)

    for i, (train_index, test_index) in enumerate(skf.split(stacked_images,
                                                            raw_labels),
                                                  1):
        trial_dst = log_dst / str(i)
        trial_dst.mkdir(exist_ok=True, parents=True)
        # split to train and test
        # xy_train, xy_test = xy_images[train_index], xy_images[test_index]
        # yz_train, yz_test = yz_images[train_index], yz_images[test_index]
        # zx_train, zx_test = zx_images[train_index], zx_images[test_index]
        image_train, image_test = stacked_images[train_index], stacked_images[test_index]
        label_train, label_test = labels[train_index], labels[test_index]
        raw_label_test = raw_labels[test_index]

        for _ in range(config["n_trial_in_ml"]):
            model = models.construct_model(input_shape=image_shape,
                                           n_class=n_class,
                                           use_model_name=args.use_model_name,
                                           n_dense_dropout=2,
                                           use_imagenet=True,
                                           show_model=False,
                                           save_model=True,
                                           filename="model",
                                           dst_dir=log_dst,
                                           extent="svg")

            # callbacks
            csv_logger = CSVLogger(trial_dst / "logger.csv")
            # early_stopping = EarlyStopping(monitor="val_loss", patience=30, mode="min")
            # f1cb = F1Callback(
            #     model, image_test,
            #     label_test)
            f1cb = F1Callback()
            history = History()
            checkpoint = ModelCheckpoint(trial_dst / "best_val_loss_model.hdf5",
                                         monitor="val_loss",
                                         save_best_only=True,
                                         save_weights_only=True)

            # compile model
            model.compile(
                loss=[
                    categorical_focal_loss(alpha=[[0.25 for _ in range(n_class)]],
                                           gamma=2)
                ],
                # metrics=["accuracy", true_positives,
                # possible_positives, predicted_positives],
                metrics=["accuracy"],
                optimizer="Adam")

            # plot_model(model, to_file="sample.png", show_shapes=True)
            # show_model(model, dst=log_dst/"model.svg", svg=True)
            # fit
            print(np.shape(image_train),
                  np.shape(image_test),
                  np.shape(label_test),
                  np.shape(label_train),
                  np.shape(weights))
            history = model.fit(
                image_train,
                label_train,
                validation_data=(image_test, label_test),
                epochs=100,
                batch_size=16,
                callbacks=[csv_logger, checkpoint],
                class_weight=weights)

            # evaluate scores at best scored model
            best_model_name = "best_val_loss_model.hdf5"
            model.load_weights(trial_dst / best_model_name)
            loss, acc, *_ = model.evaluate(
                image_test,
                label_test,
                verbose=1)
            pred_labels = np.argmax(model.predict(image_test),
                                    axis=1)

            # f1 = f1_score(np.argmax(label_test, axis=1), pred_labels, average="micro",
            #               zero_division=0)
            macrof1 = f1_score(np.argmax(label_test, axis=1), pred_labels,
                               average="macro", zero_division=0)

            if macrof1 > threshold["f1"]:
                # if complete training 3times or score good point,
                # break this for loop.
                break

        results["results"].append({
            f"trial_{i}": {
                "Accuracy": float(acc),
                "F1 score": macrof1,
            }
        })
        study_log["acc"].append(float(acc))
        # study_log["f1"].append(f1)
        study_log["f1"].append(macrof1)
        # study_log["macrof1"].append(macrof1)

        # visualizing
        visualize.visualize_history(history.history, "study_log", trial_dst)
        visualize.plot_cmx(np.argmax(label_test, axis=1),
                           pred_labels,
                           sorted_unique_label,
                           title="cmx",
                           dst=trial_dst)
        with (trial_dst / "trainlog.json").open("w") as f:
            json.dump(history.history, f, indent=2)

        with (trial_dst / "report.txt").open("w") as f:
            print(classification_report(np.argmax(label_test, axis=1),
                                        pred_labels,
                                        target_names=sorted_unique_label,
                                        zero_division=0),
                  file=f)

    with (log_dst / "weight.json").open("w") as f:
        json.dump(
            {str(k): weights[i]
             for i, k in enumerate(get_sorted_class(raw_labels))},
            f,
            indent=2)

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

    trials = ["trial_" + str(i+1) for i in range(config["use_limit"])]
    visualize.show_cmx_gif(results=[results["results"][i][t]
                                    for i, t in enumerate(trials)],
                           cmxs=[
                               log_dst / str(i+1) / "cmx.png"
                               for i in range(config["use_limit"])],
                           dst=log_dst)

    # visualize.visualize_all_cmxs(
    #     results["results"],
    #     [log_dst / str(i) / "cmx.png"
    #      for i in range(1, config["use_limit"] + 1)], results["average"], log_dst)
