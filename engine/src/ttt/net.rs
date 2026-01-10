use crate::game::{Bitboard, Game};
use crate::mcts::value_func::ValueFunction;
use crate::net::{NNetwork, ToPlanes};
use crate::ttt::{TttBitboard, TttGame, TttPosition};

impl ValueFunction<TttGame> for NNetwork<TttGame> {
    fn evaluate(&self, position: &TttPosition) -> (Vec<(<TttGame as Game>::Move, f32)>, f32) {
        self.evaluate(position)
    }
}

pub const PLANES_NUM: usize = 3;

impl ToPlanes for TttPosition {
    type Game = TttGame;

    fn to_planes(&self) -> Vec<TttBitboard> {
        let planes: [_; PLANES_NUM] = [
            // x pieces plane
            self.pieces_x(),
            // o pieces plane
            self.pieces_o(),
            // a plane with all ones to help NN find board edges
            TttBitboard::full(true),
        ];
        planes.to_vec()
    }
}
