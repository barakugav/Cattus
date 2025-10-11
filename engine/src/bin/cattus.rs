use cattus::chess::net::stockfish::StockfishNet;
use cattus::chess::uci::UCI;
use cattus::mcts::{MctsParams, TemperaturePolicy};
use cattus::net::model::InferenceConfig;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    config_file: PathBuf,
}

#[derive(serde::Deserialize)]
struct Config {
    #[allow(unused)]
    model: ModelConfig,
    mcts: MctsConfig,
    #[allow(unused)]
    threads: u32,
    seed: Option<u64>,
}
#[derive(serde::Deserialize)]
struct ModelConfig {
    #[allow(unused)]
    model_path: PathBuf,
    #[allow(unused)]
    inference: InferenceConfig,
    #[allow(unused)]
    batch_size: usize,
}
#[derive(serde::Deserialize)]
struct MctsConfig {
    sim_num: u32,
    explore_factor: f32,
    temperature_policy: Vec<(usize, f32)>,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    #[allow(unused)]
    cache_size: usize,
}

fn main() -> std::io::Result<()> {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();

    let config: Config =
        serde_json::from_reader(std::fs::File::open(&args.config_file)?).expect("failed to read config file");

    assert!(!config.mcts.temperature_policy.is_empty());
    let scheduled_temperatures = &config.mcts.temperature_policy[..config.mcts.temperature_policy.len() - 1];
    let last_temperature = config.mcts.temperature_policy.last().unwrap().1;
    let temperature = TemperaturePolicy::scheduled(scheduled_temperatures.to_vec(), last_temperature);

    let player_params = MctsParams {
        sim_num: config.mcts.sim_num,
        explore_factor: config.mcts.explore_factor,
        temperature,
        prior_noise_alpha: config.mcts.prior_noise_alpha,
        prior_noise_epsilon: config.mcts.prior_noise_epsilon,
        value_func: Arc::new(StockfishNet),
        seed: config.seed,
    };

    let mut uci = UCI::new(player_params);
    uci.run();

    Ok(())
}
