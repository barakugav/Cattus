use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use cattus::game::{Game, GameColor};
use cattus::hex::cli::{cli_print_hex_board, HexPlayerCmd};
use cattus::hex::HexGame;
use cattus::mcts::cache::ValueFuncCache;
use cattus::mcts::{MctsParams, MctsPlayer, TemperaturePolicy};

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "11")]
    board_size: u32,
    #[clap(long)]
    model_path: PathBuf,
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: usize,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
    #[clap(long, default_value = "0.0")]
    temperature_policy: String,
    #[clap(long, default_value = "0.0")]
    prior_noise_alpha: f32,
    #[clap(long, default_value = "0.0")]
    prior_noise_epsilon: f32,
    #[clap(long, default_value = "100000")]
    cache_size: usize,
}

fn run_main<const BOARD_SIZE: usize>(args: Args) {
    let mut player1 = HexPlayerCmd;

    let cache = Arc::new(ValueFuncCache::new(args.cache_size));
    let value_func = Arc::new(NNetwork::<HexGame<BOARD_SIZE>>::new(
        &args.model_path,
        InferenceConfig::default(),
        args.batch_size,
        Some(cache),
    ));
    let mut player2 = MctsPlayer::new(MctsParams {
        sim_num: args.sim_num,
        explore_factor: args.explore_factor,
        temperature: TemperaturePolicy::constant(1.0),
        prior_noise_alpha: args.prior_noise_alpha,
        prior_noise_epsilon: args.prior_noise_epsilon,
        value_func,
        seed: None,
    });

    let mut game = HexGame::<BOARD_SIZE>::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!(
        "The winner is: {}, details below:",
        match winner {
            None => String::from("Tie"),
            Some(GameColor::Player1) => String::from("CMD player"),
            Some(GameColor::Player2) => String::from("MCTS player"),
        }
    );
    cli_print_hex_board(&final_pos);
}

fn main() {
    let args = Args::parse();
    match args.board_size {
        4 => run_main::<4>(args),
        5 => run_main::<5>(args),
        7 => run_main::<7>(args),
        9 => run_main::<9>(args),
        11 => run_main::<11>(args),
        other => panic!("unsupported hex size: {other}"),
    };
}
