use cattus::chess::cli::{cli_print_chess_board, ChessPlayerCmd};
use cattus::chess::net::trivial::TrivialNet;
use cattus::chess::ChessGame;
use cattus::game::{Game, GameColor};
use cattus::mcts::{MctsParams, MctsPlayer};
use std::sync::Arc;

fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("Tie"),
        Some(GameColor::Player1) => String::from("White"),
        Some(GameColor::Player2) => String::from("Black"),
    }
}

fn main() {
    cattus::util::init_globals(Default::default());

    let mut player1 = ChessPlayerCmd;
    let value_func = Arc::new(TrivialNet);
    let mut player2 = MctsPlayer::new(MctsParams::new(10000, value_func));
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player2, &mut player1);
    println!("The winner is: {}, details below:", color_to_str(winner));
    cli_print_chess_board(&final_pos);
}
