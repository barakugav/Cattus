use cattus::game::player::{GamePlayer, PlayerRand};
use cattus::hex::uxi;
use cattus::hex::HexGameStandard;
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Player {
    Mcts,
    Rand,
}

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, value_enum, default_value = "mcts")]
    player: Player,
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: Option<usize>,
    #[clap(long)]
    model_path: Option<PathBuf>,
}

fn make_player(args: &Args) -> Box<dyn GamePlayer<HexGameStandard>> {
    match args.player {
        Player::Mcts => {
            let value_func = Arc::new(NNetwork::<HexGameStandard>::new(
                args.model_path
                    .as_ref()
                    .expect("--model_path is required for the mcts player"),
                InferenceConfig::default(),
                args.batch_size.expect("--batch_size is required for the mcts player"),
                None,
            ));
            Box::new(MctsPlayer::new(MctsParams::new(args.sim_num, value_func)))
        }
        Player::Rand => Box::new(PlayerRand::new()),
    }
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let player = make_player(&args);
    let mut engine = uxi::UxiEngine::new(player);
    engine.run();
}
