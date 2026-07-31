// Different examples pull in different games, so some of the players / board
// renderers defined here are unused per example binary.
#![allow(dead_code)]

pub mod chess;
pub mod hex;
pub mod ttt;

use cattus::game::player::GamePlayer;
use cattus::game::{Game, GameColor, GameStatus, Position};

/// A game whose board can be rendered to the terminal for the CLI examples.
pub trait CliGame: Game {
    fn print_board(pos: &Self::Position);
}

/// Play a game to completion, driving both players and narrating the game.
///
/// Before every turn the current board is printed, then the player to move is
/// asked for its move, and the chosen move is printed. The final position and
/// the winner are returned (the terminal board is left for the caller to print).
pub fn play_cli_game_until_over<G: CliGame>(
    game: &mut G,
    player1: &mut dyn GamePlayer<G>,
    player1_name: &str,
    player2: &mut dyn GamePlayer<G>,
    player2_name: &str,
) -> (G::Position, Option<GameColor>) {
    loop {
        if let GameStatus::Finished(winner) = game.status() {
            return (game.position().clone(), winner);
        }

        let positions = game.pos_history();
        let position = positions.last().unwrap();
        G::print_board(position);

        let (name, next_move) = match position.turn() {
            GameColor::Player1 => (player1_name, player1.next_move(positions).unwrap()),
            GameColor::Player2 => (player2_name, player2.next_move(positions).unwrap()),
        };
        println!("{name} plays: {next_move}");
        game.play_single_turn(next_move);
    }
}
