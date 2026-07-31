use std::io;

use cattus::game::player::GamePlayer;
use cattus::game::{Game, GameColor};
use cattus::ttt::{TttGame, TttMove, TttPosition};
use itertools::Itertools;

use super::CliGame;

pub struct TttPlayerCmd;
impl GamePlayer<TttGame> for TttPlayerCmd {
    fn next_move(&mut self, pos_history: &[TttPosition]) -> Option<TttMove> {
        let read_usize = || -> Option<usize> {
            let mut line = String::new();
            io::stdin().read_line(&mut line).expect("failed to read input");
            match line.trim().parse::<usize>() {
                Err(e) => {
                    println!("invalid number: {e}");
                    None
                }
                Ok(x) => Some(x),
            }
        };

        let position = pos_history.last().unwrap();

        loop {
            println!("Waiting for input move...");
            let Some(r) = read_usize() else { continue };
            let Some(c) = read_usize() else { continue };

            let move_ = TttMove::new(r, c);
            if position.is_valid_move(move_) {
                return Some(move_);
            }
            println!("invalid move");
        }
    }
}

impl CliGame for TttGame {
    fn print_board(pos: &TttPosition) {
        for r in 0..TttGame::BOARD_SIZE {
            let row_characters = (0..TttGame::BOARD_SIZE)
                .map(|c| match pos.get_tile(r, c) {
                    None => "_",
                    Some(GameColor::Player1) => "X",
                    Some(GameColor::Player2) => "O",
                })
                .join(" ");
            println!("{row_characters}");
        }
    }
}
