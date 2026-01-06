use clap::Parser;

use cattus::game::{Game, GameColor};
use cattus::hex::cli::{cli_print_hex_board, HexPlayerCmd};
use cattus::hex::HexGame;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "11")]
    board_size: u32,
}

fn run_main<const BOARD_SIZE: usize>() {
    let mut player1 = HexPlayerCmd;
    let mut player2 = HexPlayerCmd;

    let mut game = HexGame::<BOARD_SIZE>::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!(
        "The winner is: {}, details below:",
        match winner {
            None => "draw",
            Some(GameColor::Player1) => "player1",
            Some(GameColor::Player2) => "player2",
        }
    );
    cli_print_hex_board(&final_pos);
}

fn main() {
    let args = Args::parse();
    match args.board_size {
        3 => run_main::<3>(),
        4 => run_main::<4>(),
        5 => run_main::<5>(),
        7 => run_main::<7>(),
        9 => run_main::<9>(),
        11 => run_main::<11>(),
        other => panic!("unsupported hex size: {other}"),
    };
}
