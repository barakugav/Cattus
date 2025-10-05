pub mod player;

use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::game::player::GamePlayer;

pub trait Game: Sized {
    type Position: Position<Game = Self>;
    type Move: Move<Game = Self>;
    type Bitboard: Bitboard<Game = Self>;
    const BOARD_SIZE: usize;
    const MOVES_NUM: usize;
    const REPETITION_LIMIT: Option<usize>;

    fn new() -> Self;
    fn from_position(pos: Self::Position) -> Self;

    fn position(&self) -> &Self::Position {
        self.pos_history().last().unwrap()
    }
    fn pos_history(&self) -> &[Self::Position];

    fn status(&self) -> GameStatus;

    fn play_single_turn(&mut self, next_move: Self::Move);
    fn play_until_over(
        &mut self,
        player1: &mut impl GamePlayer<Self>,
        player2: &mut impl GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>) {
        loop {
            if let GameStatus::Finished(winner) = self.status() {
                return (self.position().clone(), winner);
            }
            let positions = self.pos_history();
            let next_move = match positions.last().unwrap().turn() {
                GameColor::Player1 => player1.next_move(positions).unwrap(),
                GameColor::Player2 => player2.next_move(positions).unwrap(),
            };
            self.play_single_turn(next_move);
        }
    }
}

pub trait Position: Clone + Eq + Hash + Debug + Send + Sync {
    type Game: Game<Position = Self>;

    fn new() -> Self;
    fn turn(&self) -> GameColor;
    fn legal_moves(&self) -> impl Iterator<Item = <Self::Game as Game>::Move>;
    fn moved_position(&self, m: <Self::Game as Game>::Move) -> Self;
    fn status(&self) -> GameStatus;
    fn flipped(&self) -> Self;
}

pub trait Move: Clone + Eq + Hash + Display + Debug + Send + Sync {
    type Game: Game<Move = Self>;

    fn flipped(&self) -> Self;
    fn to_nn_idx(&self) -> usize;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GameColor {
    Player1,
    Player2,
}
impl GameColor {
    pub fn opposite(&self) -> GameColor {
        match self {
            GameColor::Player1 => GameColor::Player2,
            GameColor::Player2 => GameColor::Player1,
        }
    }

    pub fn to_signed_one(player: Option<GameColor>) -> i32 {
        match player {
            Some(GameColor::Player1) => 1,
            Some(GameColor::Player2) => -1,
            None => 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GameStatus {
    Ongoing,
    Finished(Option<GameColor>),
}
impl GameStatus {
    pub fn is_ongoing(&self) -> bool {
        matches!(self, GameStatus::Ongoing)
    }
    pub fn is_finished(&self) -> bool {
        matches!(self, GameStatus::Finished(_))
    }
}

pub trait Bitboard: Clone {
    type Game: Game<Bitboard = Self>;

    fn new() -> Self;
    fn full(value: bool) -> Self;
    fn get(&self, idx: usize) -> bool;
    fn set(&mut self, idx: usize, val: bool);
}
