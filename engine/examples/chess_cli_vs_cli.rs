use cattus::chess::cli::{cli_print_chess_board, ChessPlayerCmd};
use cattus::chess::ChessGame;
use cattus::game::{Game, GameColor};

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
    let mut player2 = ChessPlayerCmd;
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    cli_print_chess_board(&final_pos);
}
