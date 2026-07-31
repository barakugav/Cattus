import random

import numpy as np
from cattus_train.chess import NN_INDEX_TO_MOVE, Chess
from cattus_train.data_set import DataSet
from cattus_train.trainable_game import DataEntry


def _make_ds():
    ds = DataSet.__new__(DataSet)
    ds._game = Chess()
    return ds


def _make_entry():
    # no castle rights / no pawns -> every mirror branch is allowed
    planes = np.zeros((Chess.PLANES_NUM, Chess.BOARD_SIZE, Chess.BOARD_SIZE), dtype=np.float32)
    probs = np.full((Chess.MOVE_NUM,), -1.0, dtype=np.float32)
    probs[0] = 1.0
    return DataEntry(planes=planes, probs=probs, winner=0.0)


def test_chess_augmentation_does_not_corrupt_global_move_table():
    ds = _make_ds()
    before = [m.uci() for m in NN_INDEX_TO_MOVE]

    orig = random.random
    random.random = lambda: 0.0  # force all mirror branches
    try:
        idx_a = int(np.argmax(_transform(ds).probs))
        idx_b = int(np.argmax(_transform(ds).probs))
    finally:
        random.random = orig

    assert [m.uci() for m in NN_INDEX_TO_MOVE] == before  # globals untouched
    assert idx_a == idx_b  # identical input + augmentation -> identical target


def _transform(ds):
    entry = _make_entry()
    ds._transform_chess(entry)
    return entry
