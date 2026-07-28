import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cattus_train.chess import Chess
from cattus_train.config import TrainingConfig
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import DataEntry, Game


class DataSet(Dataset):
    def __init__(self, game: Game, train_data_dir: Path, cfg: TrainingConfig, device: torch.device):
        self._game: Game = game
        self._train_data_dir: Path = train_data_dir
        self._cfg = cfg
        self._device: torch.device = device

        self._filenames = self._data_entries_filenames()

    def _data_entries_filenames(self) -> list[Path]:
        filenames = [p for p in self._train_data_dir.rglob("*.traindata")]

        # take the latests files
        filenames.sort(key=os.path.getmtime, reverse=True)
        filenames = filenames[: self._cfg.latest_data_entries]

        return filenames

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, index: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        filename = self._filenames[index]
        packed_entry = self._game.load_data_entry(filename)
        entry = DataSet.unpack_planes(packed_entry, self._game)

        # Transform (data augmentation)
        self.transform(entry)

        planes = torch.tensor(np.ascontiguousarray(entry.planes), dtype=torch.float32, device=self._device)
        probs = torch.tensor(np.ascontiguousarray(entry.probs), dtype=torch.float32, device=self._device)
        winner = torch.tensor(entry.winner, dtype=torch.float32, device=self._device)
        return planes, (probs, winner)

    @staticmethod
    def unpack_planes(packed_entry: DataEntry, game: Game) -> DataEntry:
        # planes, (probs, winner) = packed_entry
        assert len(packed_entry.planes) == game.PLANES_NUM
        plane_size = game.BOARD_SIZE * game.BOARD_SIZE
        planes = [np.frombuffer(plane, dtype=np.uint8) for plane in packed_entry.planes]
        planes = np.array([np.unpackbits(plane, count=plane_size, bitorder="little") for plane in planes])
        planes = planes.reshape((game.PLANES_NUM, game.BOARD_SIZE, game.BOARD_SIZE))
        return DataEntry(planes=planes, probs=packed_entry.probs, winner=packed_entry.winner)

    def transform(self, entry: DataEntry):
        match self._game:
            case TicTacToe():
                self._transform_ttt(entry)
            case Hex():
                self._transform_hex(entry)
            case Chess():
                self._transform_chess(entry)
            case _:
                raise NotImplementedError(f"DataSet.transform not implemented for game {self._game}")

    def _transform_ttt(self, entry: DataEntry):
        planes, probs = entry.planes, entry.probs
        probs = probs.reshape((3, 3))

        # Use all combination of the basic transforms:
        # - original
        # - rows mirror
        # - columns mirror
        # - diagonal mirror
        # - rows + columns = rotate 180
        # - rows + diagonal = rotate 90
        # - columns + diagonal = rotate 90 (other direction)
        # - row + columns + diagonal = other diagonal mirror

        # Rows mirror
        if random.random() < 0.5:
            planes = np.flip(planes, axis=1)
            probs = np.flip(probs, axis=0)

        # Columns mirror
        if random.random() < 0.5:
            planes = np.flip(planes, axis=2)
            probs = np.flip(probs, axis=1)

        # Diagonal mirror
        if random.random() < 0.5:
            planes = np.transpose(planes, (0, 2, 1))
            probs = np.transpose(probs, (1, 0))

        entry.planes = planes
        entry.probs = probs.flatten()

    def _transform_hex(self, entry: DataEntry):
        planes, probs = entry.planes, entry.probs
        probs = probs.reshape((self._game.BOARD_SIZE, self._game.BOARD_SIZE))

        # Rotate 180
        if random.random() < 0.5:
            planes = np.flip(planes, axis=(1, 2))
            probs = np.flip(probs, axis=(0, 1))

        entry.planes = planes
        entry.probs = probs.flatten()

    def _transform_chess(self, entry: DataEntry):
        planes = entry.planes
        probs = entry.probs

        import chess

        def probs_as_list(probs) -> list[tuple[chess.Move, float]]:
            if isinstance(probs, list):
                return probs
            from cattus_train.chess import NN_INDEX_TO_MOVE

            assert isinstance(probs, np.ndarray)
            # copy moves so the shared NN_INDEX_TO_MOVE objects are never mutated below
            return [(chess.Move(m.from_square, m.to_square, promotion=m.promotion), p) for m, p in zip(NN_INDEX_TO_MOVE, probs)]

        def probs_as_array(probs) -> np.ndarray:
            if isinstance(probs, np.ndarray):
                return probs
            from cattus_train.chess import MOVE_TO_NN_INDEX, chess_move_to_idx

            assert isinstance(probs, list)
            arr = np.zeros(len(probs), dtype=np.float32)
            for move, p in probs:
                arr[MOVE_TO_NN_INDEX[chess_move_to_idx(move)]] = p
            return arr

        ### Planes
        # 0 - white pawns
        # 1 - white knights
        # 2 - white bishops
        # 3 - white rooks
        # 4 - white queens
        # 5 - white kings
        # 6 - black pawns
        # 7 - black knights
        # 8 - black bishops
        # 9 - black rooks
        # 10 - black queens
        # 11 - black kings
        # 12 - white can castle kingside
        # 13 - white can castle queenside
        # 14 - black can castle kingside
        # 15 - black can castle queenside
        # 16 - en passant square
        # 17 - all ones
        has_castle_rights = planes[12, 0, 0] or planes[13, 0, 0] or planes[14, 0, 0] or planes[15, 0, 0]
        has_pawns = planes[0].any() or planes[6].any()

        # Use all combination of the basic transforms:
        # - original
        # - rows mirror
        # - columns mirror
        # - diagonal mirror
        # - rows + columns = rotate 180
        # - rows + diagonal = rotate 90
        # - columns + diagonal = rotate 90 (other direction)
        # - row + columns + diagonal = other diagonal mirror

        # Rows mirror
        if not has_castle_rights and not has_pawns and random.random() < 0.5:
            planes = np.flip(planes, axis=1)
            probs = probs_as_list(probs)
            for move, _p in probs:
                # flip rank
                move.from_square ^= 0b111000
                move.to_square ^= 0b111000

        # Columns mirror
        if not has_castle_rights and random.random() < 0.5:
            planes = np.flip(planes, axis=2)
            probs = probs_as_list(probs)
            for move, _p in probs:
                # flip file
                move.from_square ^= 0b000111
                move.to_square ^= 0b000111

        # Diagonal mirror
        if not has_castle_rights and not has_pawns and random.random() < 0.5:
            planes = np.transpose(planes, (0, 2, 1))
            probs = probs_as_list(probs)
            for move, _p in probs:
                src, dst = move.from_square, move.to_square
                move.from_square = chess.square(chess.square_rank(src), chess.square_file(src))
                move.to_square = chess.square(chess.square_rank(dst), chess.square_file(dst))

        entry.planes = planes
        entry.probs = probs_as_array(probs)
