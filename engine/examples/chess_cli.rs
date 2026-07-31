mod common;

#[cfg(feature = "stockfish")]
use cattus::chess::net::stockfish::StockfishNet;
use cattus::chess::net::trivial::TrivialNet;
use cattus::chess::ChessGame;
use cattus::game::player::GamePlayer;
use cattus::game::{Game, GameColor};
use cattus::mcts::cache::ValueFuncCache;
use cattus::mcts::{MctsParams, MctsPlayer, TemperaturePolicy};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use clap::Parser;
use common::chess::ChessPlayerCmd;
use common::CliGame;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Player {
    Cli,
    Mcts,
    Trivial,
    #[cfg(feature = "stockfish")]
    Stockfish,
}

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
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

fn color_to_str(c: Option<GameColor>) -> &'static str {
    match c {
        None => "Tie",
        Some(GameColor::Player1) => "White",
        Some(GameColor::Player2) => "Black",
    }
}

fn make_player(player: Player, args: &Args) -> Box<dyn GamePlayer<ChessGame>> {
    match player {
        Player::Cli => Box::new(ChessPlayerCmd),
        Player::Mcts => {
            let cache = Arc::new(ValueFuncCache::new(args.cache_size));
            let value_func = Arc::new(NNetwork::<ChessGame>::new(
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
        Player::Trivial => Box::new(MctsPlayer::new(MctsParams::new(10000, Arc::new(TrivialNet)))),
        #[cfg(feature = "stockfish")]
        Player::Stockfish => Box::new(MctsPlayer::new(MctsParams::new(100000, Arc::new(StockfishNet)))),
    }
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let mut player1 = make_player(args.player1, &args);
    let mut player2 = make_player(args.player2, &args);
    let mut game = ChessGame::new();

    let (final_pos, winner) =
        common::play_cli_game_until_over(&mut game, &mut *player1, "White", &mut *player2, "Black");
    println!("The winner is: {}, details below:", color_to_str(winner));
    ChessGame::print_board(&final_pos);
}
