use rand::prelude::*;

use crate::game::Position;

pub trait GamePlayer<Game: crate::game::Game> {
    fn next_move(&mut self, pos_history: &[Game::Position]) -> Option<Game::Move>;
}

pub struct PlayerRand {
    rand: StdRng,
}
impl Default for PlayerRand {
    fn default() -> Self {
        Self::new()
    }
}
impl PlayerRand {
    pub fn new() -> Self {
        Self::from_seed(rand::rng().random())
    }

    pub fn from_seed(seed: u64) -> Self {
        Self {
            rand: StdRng::seed_from_u64(seed),
        }
    }
}

impl<Game: crate::game::Game> GamePlayer<Game> for PlayerRand {
    fn next_move(&mut self, pos_history: &[Game::Position]) -> Option<Game::Move> {
        let moves = pos_history.last().unwrap().legal_moves().collect::<Vec<_>>();
        moves.choose(&mut self.rand).cloned()
    }
}
