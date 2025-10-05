pub mod chess;
pub mod hex;
pub mod ttt;

use cattus::game::{GameColor, Move};

pub struct DataEntry<Game: cattus::game::Game> {
    pub pos: Game::Position,
    pub probs: Vec<(Game::Move, f32)>,
    pub winner: Option<GameColor>,
}

pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

pub fn generic_entry_to_bytes<Game: cattus::game::Game>(
    planes: &[u64],
    probs: &[(Game::Move, f32)],
    winner: i8,
) -> Vec<u8> {
    /* Use -1 for illegal moves */
    let mut probs_vec = vec![-1.0f32; Game::MOVES_NUM];

    /* Fill legal moves probabilities */
    for (m, prob) in probs {
        probs_vec[m.to_nn_idx()] = *prob;
    }

    let u64bytes = u64::BITS as usize / 8;
    let f32bytes = /* f32::BITS */ 32 / 8;
    let i8bytes = i8::BITS as usize / 8;
    let size = planes.len() * u64bytes + probs_vec.len() * f32bytes + i8bytes;
    let mut bytes = Vec::with_capacity(size);

    /* Serialized in little indian format, should deserialized the same */
    bytes.extend(planes.iter().flat_map(|p| p.to_le_bytes()));
    bytes.extend(probs_vec.into_iter().flat_map(|p| p.to_le_bytes()));
    bytes.extend(winner.to_le_bytes());
    assert_eq!(bytes.len(), size);

    bytes
}
