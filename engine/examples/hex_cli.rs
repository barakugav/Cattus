mod common;

use cattus::game::player::GamePlayer;
use cattus::game::{Game, GameColor};
use cattus::hex::HexGame;
use cattus::mcts::cache::ValueFuncCache;
use cattus::mcts::{MctsParams, MctsPlayer, TemperaturePolicy};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use clap::Parser;
use common::hex::HexPlayerCmd;
use common::CliGame;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Player {
    Cli,
    Mcts,
}

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "11")]
    board_size: u32,
    #[clap(long, value_enum, default_value = "cli")]
    player1: Player,
    #[clap(long, value_enum, default_value = "mcts")]
    player2: Player,
    #[clap(long)]
    model_path: Option<PathBuf>,
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: Option<usize>,
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

fn make_player<const BOARD_SIZE: usize>(player: Player, args: &Args) -> Box<dyn GamePlayer<HexGame<BOARD_SIZE>>> {
    match player {
        Player::Cli => Box::new(HexPlayerCmd),
        Player::Mcts => {
            let cache = Arc::new(ValueFuncCache::new(args.cache_size));
            let value_func = Arc::new(NNetwork::<HexGame<BOARD_SIZE>>::new(
                args.model_path
                    .as_ref()
                    .expect("--model_path is required for the mcts player"),
                InferenceConfig::default(),
                args.batch_size.expect("--batch_size is required for the mcts player"),
                Some(cache),
            ));
            Box::new(MctsPlayer::new(MctsParams {
                sim_num: args.sim_num,
                explore_factor: args.explore_factor,
                temperature: TemperaturePolicy::constant(1.0),
                prior_noise_alpha: args.prior_noise_alpha,
                prior_noise_epsilon: args.prior_noise_epsilon,
                value_func,
                seed: None,
            }))
        }
    }
}

fn run_main<const BOARD_SIZE: usize>(args: Args) {
    let mut player1 = make_player::<BOARD_SIZE>(args.player1, &args);
    let mut player2 = make_player::<BOARD_SIZE>(args.player2, &args);
    let mut game = HexGame::<BOARD_SIZE>::new();

    let (final_pos, winner) =
        common::play_cli_game_until_over(&mut game, &mut *player1, "player1", &mut *player2, "player2");
    println!(
        "The winner is: {}, details below:",
        match winner {
            None => "draw",
            Some(GameColor::Player1) => "player1",
            Some(GameColor::Player2) => "player2",
        }
    );
    HexGame::<BOARD_SIZE>::print_board(&final_pos);
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();
    match args.board_size {
        3 => run_main::<3>(args),
        4 => run_main::<4>(args),
        5 => run_main::<5>(args),
        7 => run_main::<7>(args),
        9 => run_main::<9>(args),
        11 => run_main::<11>(args),
        other => panic!("unsupported hex size: {other}"),
    };
}
