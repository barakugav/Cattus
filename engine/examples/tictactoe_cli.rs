mod common;

use cattus::game::player::GamePlayer;
use cattus::game::Game;
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use cattus::ttt::{color_to_str, TttGame};
use clap::Parser;
use common::ttt::TttPlayerCmd;
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
    #[clap(long, value_enum, default_value = "cli")]
    player1: Player,
    #[clap(long, value_enum, default_value = "mcts")]
    player2: Player,
    #[clap(long)]
    model_path: Option<PathBuf>,
    #[clap(long)]
    batch_size: Option<usize>,
}

fn make_player(player: Player, args: &Args) -> Box<dyn GamePlayer<TttGame>> {
    match player {
        Player::Cli => Box::new(TttPlayerCmd),
        Player::Mcts => {
            let value_func = Arc::new(NNetwork::<TttGame>::new(
                args.model_path
                    .as_ref()
                    .expect("--model_path is required for the mcts player"),
                InferenceConfig::default(),
                args.batch_size.expect("--batch_size is required for the mcts player"),
                None,
            ));
            Box::new(MctsPlayer::new(MctsParams::new(1000, value_func)))
        }
    }
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let mut player1 = make_player(args.player1, &args);
    let mut player2 = make_player(args.player2, &args);
    let mut game = TttGame::new();

    let (final_pos, winner) = common::play_cli_game_until_over(&mut game, &mut *player1, "X", &mut *player2, "O");
    println!("The winner is: {}, details below:", color_to_str(winner));
    TttGame::print_board(&final_pos);
}
