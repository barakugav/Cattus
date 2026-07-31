use std::io;

use cattus::game::player::GamePlayer;
use cattus::game::{Game, GameColor};
use cattus::hex::{HexGame, HexMove, HexPosition};

use super::CliGame;

pub struct HexPlayerCmd;
impl<const BOARD_SIZE: usize> GamePlayer<HexGame<BOARD_SIZE>> for HexPlayerCmd {
    fn next_move(
        &mut self,
        pos_history: &[<HexGame<BOARD_SIZE> as Game>::Position],
    ) -> Option<<HexGame<BOARD_SIZE> as Game>::Move> {
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
            let m = HexMove::new(r, c);

            if position.is_valid_move(m) {
                return Some(m);
            }
            println!("invalid move");
        }
    }
}

impl<const BOARD_SIZE: usize> CliGame for HexGame<BOARD_SIZE> {
    fn print_board(pos: &HexPosition<BOARD_SIZE>) {
        for r in 0..BOARD_SIZE {
            let row_characters: Vec<String> = (0..BOARD_SIZE)
                .map(|c| {
                    String::from(match pos.get_tile(r, c) {
                        None => '·',
                        Some(GameColor::Player1) => 'R',
                        Some(GameColor::Player2) => 'B',
                    })
                })
                .collect();
            let spaces = " ".repeat(r);
            println!("{}{}", spaces, row_characters.join(" "));
        }
    }
}
