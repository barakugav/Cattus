use crate::game::Bitboard;
use crate::hex::{HexBitboard, HexGame, HexMove, HexPosition};
use crate::mcts::value_func::ValueFunction;
use crate::net::{NNetwork, ToPlanes};

impl<const BOARD_SIZE: usize> ValueFunction<HexGame<BOARD_SIZE>> for NNetwork<HexGame<BOARD_SIZE>> {
    fn evaluate(&self, position: &HexPosition<BOARD_SIZE>) -> (Vec<(HexMove<BOARD_SIZE>, f32)>, f32) {
        self.evaluate(position)
    }
}

pub const PLANES_NUM: usize = 3;

impl<const BOARD_SIZE: usize> ToPlanes for HexPosition<BOARD_SIZE> {
    type Game = HexGame<BOARD_SIZE>;

    fn to_planes(&self) -> Vec<HexBitboard<BOARD_SIZE>> {
        let planes: [_; PLANES_NUM] = [
            /* red pieces plane */
            self.pieces_red(),
            /* blue pieces plane */
            self.pieces_blue(),
            /* a plane with all ones to help NN find board edges */
            HexBitboard::full(true),
        ];
        planes.to_vec()
    }
}
