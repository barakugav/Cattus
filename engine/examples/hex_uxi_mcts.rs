use cattus::hex::uxi;
use cattus::hex::HexGameStandard;
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: usize,
    #[clap(long)]
    model_path: PathBuf,
}

fn main() {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let value_func = Arc::new(NNetwork::<HexGameStandard>::new(
        &args.model_path,
        InferenceConfig::default(),
        args.batch_size,
        None,
    ));
    let player = Box::new(MctsPlayer::new(MctsParams::new(args.sim_num, value_func)));
    let mut engine = uxi::UxiEngine::new(player);
    engine.run();
}
