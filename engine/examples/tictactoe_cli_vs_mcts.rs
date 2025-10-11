use cattus::game::Game;
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use cattus::ttt::cli::{cli_print_ttt_board, TttPlayerCmd};
use cattus::ttt::{color_to_str, TttGame};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: PathBuf,
    #[clap(long)]
    batch_size: usize,
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let value_func = Arc::new(NNetwork::new(
        &args.model_path,
        InferenceConfig::default(),
        args.batch_size,
        None,
    ));
    let mut player1 = MctsPlayer::new(MctsParams::new(1000, value_func));

    let mut player2 = TttPlayerCmd;

    let mut game = TttGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    cli_print_ttt_board(&final_pos);
}
