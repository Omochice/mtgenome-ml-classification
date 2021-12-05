import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.applications import (DenseNet121, EfficientNetB0,
                                           EfficientNetB1, EfficientNetB2,
                                           MobileNet, MobileNetV2,
                                           MobileNetV3Small, NASNetMobile,
                                           ResNet50, ResNet50V2)
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Dropout, Input
from tensorflow.keras.utils import model_to_dot, plot_model

Openable = Union[str, Path]

models = {
    "mobilenet": MobileNet,
    "mobilenetV2": MobileNetV2,
    "mobilenetV3": MobileNetV3Small,
    "resnet": ResNet50,
    "resnetV2": ResNet50V2,
    "densenet": DenseNet121,
    "efficientB0": EfficientNetB0,
    "efficientB1": EfficientNetB1,
    "efficientB2": EfficientNetB2,
    "nasnetmobile": NASNetMobile,
}

Extent = Literal["svg", "png"]


def fix_svg_s_scaling(path: Openable) -> None:
    """Fix svg image scaling.
    If specify `svg` in `model_to_dot`, generated image is scale(0.75 0.75)
    This function fix there scale to `(1.0 1.0)`.

    Args:
        path(Openable): The path to SVG image

    Returns:
        None:
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if len(root) >= 1 and "transform" in root[0].attrib:
        original = root[0].attrib["transform"]
        fixed = re.sub(r"scale\(\d+\.?\d*\s\d+\.?\d*\)", "scale(1.0 1.0)", original)
        root[0].attrib["transform"] = fixed
    tree.write(path)


def construct_model(
    input_shape: Tuple[int, int, int],
    n_class: int,
    use_model_name: str,
    parallel: bool = False,
    n_dense_dropout: int = 2,
    use_imagenet: bool = True,
    show_model: bool = False,
    save_model: bool = True,
    filename: str = "model",
    dst_dir: Optional[Openable] = None,
    extent: str = "png",
) -> tf.keras.Model:
    f"""Construct model.

    Args:
        input_shape (Tuple[int, int, int]): Input shape.
                                           (height, width, depth), depth must be 1 or 3.
        n_class (int): Number of class to classification.
        use_model_name (str): Use model name. It must be in {models.keys()}.
        parallel (bool): Train each dimention image in different CNN. default to False.
        n_dense_dropout (int): Number of Dense-Dropout layser set. default to 2.
        use_imagenet (bool): Whether use imagenet weights.
        show_model (bool): Whether show model shape in STDOUT.
        save_model (bool): Whether save model as image.
        filename (str): filename when save model as image.
        dst_dir (Openable): save directory name when save model as image.
        extent (str): Exntent when save model as image.

    Returns:
        tf.keras.Model:
    """

    assert input_shape[-1] in {1, 3}
    assert use_model_name in models

    weights: Optional[str] = None
    if use_imagenet:
        weights = "imagenet"

    if parallel:
        s = (input_shape[0], input_shape[1], 1)
        input_x_y = Input(shape=s)
        input_y_z = Input(shape=s)
        input_z_x = Input(shape=s)

        x_y = Conv2D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(input_x_y)
        y_z = Conv2D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(input_y_z)
        z_x = Conv2D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(input_z_x)
        xy = models[use_model_name](
            include_top=False,
            input_shape=(*input_shape[:2], 3),
            weights=weights,
            pooling="max",
        )
        xy._name = f"{use_model_name}_xy"
        yz = models[use_model_name](
            include_top=False,
            input_shape=(*input_shape[:2], 3),
            weights=weights,
            pooling="max",
        )
        yz._name = f"{use_model_name}_yz"
        zx = models[use_model_name](
            include_top=False,
            input_shape=(*input_shape[:2], 3),
            weights=weights,
            pooling="max",
        )
        zx._name = f"{use_model_name}_zx"
        x_y = xy(x_y)
        y_z = yz(y_z)
        z_x = zx(z_x)
        model = Concatenate()([x_y, y_z, z_x])
        for i in range(n_dense_dropout):
            model = Dense(512, activation="relu")(model)
            model = Dropout(0.3)(model)
        model = Dense(n_class, activation="softmax")(model)
        constructed = tf.keras.Model(
            inputs=[input_x_y, input_y_z, input_z_x],
            outputs=model,
        )
    else:
        input_ = Input(shape=input_shape)
        if input_shape[2] == 3:
            model = models[use_model_name](
                include_top=False,
                input_shape=(*input_shape[:2], 3),
                weights=weights,
                pooling="max",
            )(input_)
        else:
            model = Conv2D(filters=3, kernel_size=3, padding="same")(input_)
            model = models[use_model_name](
                include_top=False,
                input_shape=(*input_shape[:2], 3),
                weights=weights,
                pooling="max",
            )(model)

        for i in range(n_dense_dropout):
            model = Dense(512, activation="relu")(model)
            model = Dropout(0.3)(model)
        model = Dense(n_class, activation="softmax")(model)

        constructed = tf.keras.Model(inputs=input_, outputs=model)

    if show_model:
        constructed.summary()

    if save_model:
        dst = Path(dst_dir) / (filename + "." + extent)
        plot_model(
            constructed,
            to_file=dst,
            show_dtype=False,
            show_layer_names=False,
            show_shapes=True,
        )
        if extent == "svg":
            fix_svg_s_scaling(dst)

    return constructed


# def construct_parallel_model(
#     n_class: int, input_shape: Tuple[int, int, int]
# ) -> tf.keras.Model:
#     """construct_model.
# 
#     Args:
#         n_class (int): n_class
#         shape (Tuple[int, int]): shape
# 
#     Returns:
#         tf.keras.Model:
#     """
#     shape = input_shape[:2]
#     input_x_y = Input(shape=(*shape, 1))
#     input_y_z = Input(shape=(*shape, 1))
#     input_z_x = Input(shape=(*shape, 1))
# 
#     x_y = Conv2D(
#         filters=3,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#     )(input_x_y)
#     y_z = Conv2D(
#         filters=3,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#     )(input_y_z)
#     z_x = Conv2D(
#         filters=3,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#     )(input_z_x)
#     mobilenet_xy = MobileNetV2(
#         include_top=False,
#         input_shape=(*shape, 3),
#         weights="imagenet",
#         pooling="max",
#     )
#     mobilenet_xy._name = "mobilenetv2_xy_dim"
#     mobilenet_yz = MobileNetV2(
#         include_top=False,
#         input_shape=(*shape, 3),
#         weights="imagenet",
#         pooling="max",
#     )
#     mobilenet_yz._name = "mobilenetv2_yz_dim"
#     mobilenet_zx = MobileNetV2(
#         include_top=False,
#         input_shape=(*shape, 3),
#         weights="imagenet",
#         pooling="max",
#     )
#     mobilenet_zx._name = "mobilenetv2_zx_dim"
#     x_y = mobilenet_xy(x_y)
#     y_z = mobilenet_yz(y_z)
#     z_x = mobilenet_zx(z_x)
#     concate = Concatenate()([x_y, y_z, z_x])
#     model = Dense(512, activation="relu")(concate)
#     model = Dropout(0.3)(model)
#     model = Dense(512, activation="relu")(model)
#     model = Dropout(0.25)(model)
#     model = Dense(n_class, activation="softmax")(model)
# 
#     return tf.keras.Model(
#         inputs=[input_x_y, input_y_z, input_z_x],
#         outputs=model,
#     )


# m = construct_model(
#     (128, 128, 3),
#     52,
#     "efficientB0",
#     show_model=True,
#     save_model=True,
#     dst_dir=".",
#     extent="svg",
#     filename="samplele"
# )
