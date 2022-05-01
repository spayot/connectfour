# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = puissance4.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import copy
import os
from threading import Lock

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import board_config, pvn_config
from .game import ConnectFourGameState

tf.get_logger().setLevel("ERROR")


def softmax(x: np.ndarray) -> np.ndarray:
    """applies softmax to an array"""
    m = np.max(x)
    probs = np.exp(x - m)
    probs /= np.sum(probs)
    return probs


def crossentropy_loss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10
) -> tf.Tensor:
    return -K.mean(K.sum(y_true * K.log(y_pred + eps), axis=1))


def conv_block(input_tensor, kernel_size, filter, l2_const):
    x = input_tensor
    x = layers.Conv2D(
        filter,
        kernel_size,
        padding="same",
        kernel_regularizer=l2(l2_const),
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    out = layers.Activation("relu")(x)
    return out


def residual_block(input_tensor, kernel_size, filter, l2_const):

    shortcut = layers.Conv2D(
        filter,
        kernel_size=(1, 1),
        padding="same",
        kernel_regularizer=l2(l2_const),
        kernel_initializer="he_normal",
    )(input_tensor)

    x = input_tensor
    x = layers.Conv2D(
        filter,
        kernel_size,
        padding="same",
        kernel_regularizer=l2(l2_const),
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filter,
        kernel_size,
        padding="same",
        kernel_regularizer=l2(l2_const),
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    out = layers.Activation("relu")(x)
    return out


class PolicyValueNet:
    def __init__(self, n: int = 6, m: int = 7, name: str = None, quiet: bool = True):
        self.n = n
        self.m = m
        self.name = name
        self.l2_const = pvn_config["l2_const"]
        self.pvnet_fn_lock = Lock()

        self.build_model()

        # if filename != None and os.path.exists(filename):
        #     self.model.load_weights(filename)
        #     self.name = os.path.split(filename)[-1].split('.')[0]

        if quiet:
            print("To see model details, enter:\n\t>>> <pvn>.summary()" "")
        else:
            print(self.model.summary())

    @classmethod
    def from_file(cls, filename: str):
        pvn = cls()
        pvn.model.load_weights(filename)
        pvn.name = _extract_model_name_from_file(filename)
        return pvn

    def build_model(self) -> None:
        x = net = Input((self.n, self.m, 1))

        net = conv_block(net, (3, 3), 128, self.l2_const)
        for _ in range(pvn_config["block_size"]):
            net = residual_block(net, (3, 3), 128, self.l2_const)

        policy_net = layers.Conv2D(
            filters=2, kernel_size=(1, 1), kernel_regularizer=l2(self.l2_const)
        )(net)
        policy_net = layers.BatchNormalization()(policy_net)
        policy_net = layers.Activation("relu")(policy_net)
        policy_net = layers.Flatten()(policy_net)
        self.policy_net = layers.Dense(
            self.m,
            activation="softmax",
            kernel_regularizer=l2(self.l2_const),
            name="policy_head",
        )(policy_net)

        value_net = layers.Conv2D(
            filters=1, kernel_size=(1, 1), kernel_regularizer=l2(self.l2_const)
        )(net)
        value_net = layers.BatchNormalization()(value_net)
        value_net = layers.Activation("relu")(value_net)
        value_net = layers.Flatten()(value_net)
        value_net = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(self.l2_const),
        )(value_net)
        self.value_net = layers.Dense(
            1,
            activation="tanh",
            kernel_regularizer=l2(self.l2_const),
            name="value_head",
        )(value_net)

        self.model = Model(x, [self.policy_net, self.value_net], name=self.name)

    def get_train_fn(self):
        losses = [crossentropy_loss, "mean_squared_error"]
        self.model.compile(optimizer=Adam(lr=0.002, eps=1e-6), loss=losses)

        batch_size = pvn_config["batch_size"]
        epochs = pvn_config["epochs"]

        def train_fn(input_boards, policy, value):
            history = self.model.fit(
                np.asarray(input_boards),
                [np.asarray(policy), np.asarray(value)],
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
            )
            print("train history:", history.history)

        return train_fn

    def infer_from_state(
        self, state: ConnectFourGameState
    ) -> "tuple[np.ndarray, float]":
        # self.pvnet_fn_lock.acquire()
        # with self.graph.as_default():
        probs, value = self.model.predict(
            state.board.reshape(1, self.n, self.m, 1) * state.next_player.value
        )
        # self.pvnet_fn_lock.release()

        return probs[0], value[0][0] * state.next_player.value

    def evaluate_state(self, state: ConnectFourGameState) -> "tuple[np.ndarray, float]":
        """Policy and Value estimations based on the current state.
        Arguments:
            - state: current state

        Returns:
        - policy: 7x1 array providing a probability distribution of each move being the best move for the player due to play.
        - value: expected outcome of the game (1 for a win of player 1, -1 for a win of player 2, 0 for a draw)

        Example:
        >>> evaluator = Evaluator(name='test')
        >>> evaluator.evaluate_state(ConnectFourGameState(board=np.zeros(6,7), next_to_move=1))"""

        if state.is_game_over:
            p, v = np.ones(7) / 7, state.game_result

        else:
            p, v = self.infer_from_state(state)

        return p, v

    def save_model(self, model_dir: str = "models/") -> None:
        fname = os.path.join(model_dir, self.name + ".h5")
        if os.path.exists(fname):
            os.remove(fname)
        # self.model.save(model_file)
        self.model.save_weights(fname)

    def __repr__(self) -> str:
        return __class__.__name__ + f"(name={self.name})"


def _extract_model_name_from_file(filename: str):
    return os.path.split(filename)[-1].split(".")[0]
