pub mod model;

use crate::game::{Bitboard, GameColor, Move, Position};
use crate::mcts::cache::ValueFuncCache;
use crate::util::batch::Batcher;
use crate::util::metric::RunningAverage;
use itertools::Itertools;
use model::{InferenceConfig, Model};
use ndarray::{Array2, Array4};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct NNetwork<Game: crate::game::Game> {
    model: Mutex<Model>,
    cache: Option<Arc<ValueFuncCache<Game>>>,

    batcher: Batcher<Vec<Game::Bitboard>, (Vec<f32>, f32)>,

    metrics: Mutex<Metrics>,
}

impl<Game: crate::game::Game> NNetwork<Game> {
    pub fn new(
        model_path: impl AsRef<Path>,
        inference_cfg: InferenceConfig,
        batch_size: usize,
        cache: Option<Arc<ValueFuncCache<Game>>>,
    ) -> Self {
        Self {
            model: Mutex::new(Model::new(model_path, inference_cfg)),
            cache,
            batcher: Batcher::new(batch_size),
            metrics: Mutex::new(Metrics {
                activation_count: metrics::counter!("model.activation_count"),
                run_duration: RunningAverage::new(0.99, metrics::gauge!("model.run_duration")),
            }),
        }
    }

    pub fn run_net(&self, input: Array4<f32>) -> Vec<(Vec<f32>, f32)> {
        let net_run_begin = Instant::now();
        let outputs = self.model.lock().unwrap().run([input.into_dyn()].to_vec());
        let run_duration = net_run_begin.elapsed();

        let (moves_scores, vals) = outputs.into_iter().collect_tuple().unwrap();
        let moves_scores: Array2<f32> = moves_scores.into_dimensionality().unwrap();
        let vals: Array2<f32> = vals.into_dimensionality().unwrap();

        // let batch_size = vals.len();
        let ret = moves_scores
            .rows()
            .into_iter()
            .zip(vals)
            .map(|(sample_scores, val)| {
                let mut sample_scores = sample_scores.to_vec();
                for s in sample_scores.iter_mut() {
                    if !s.is_finite() {
                        *s = f32::MIN;
                    }
                }
                (sample_scores, val)
            })
            .collect_vec();

        // update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.activation_count.increment(1);
        metrics.run_duration.set(run_duration.as_secs_f64());

        ret
    }

    pub fn evaluate(
        &self,
        position: &Game::Position,
        to_planes: impl Fn(&Game::Position) -> Vec<Game::Bitboard>,
    ) -> (Vec<(Game::Move, f32)>, f32) {
        let (position, is_flipped) = flip_pos_if_needed(position.clone());

        let res = if let Some(cache) = &self.cache {
            cache.get_or_compute(&position, |pos| self.evaluate_impl(pos, &to_planes))
        } else {
            self.evaluate_impl(&position, &to_planes)
        };

        flip_score_if_needed(res, is_flipped)
    }
    fn evaluate_impl(
        &self,
        pos: &Game::Position,
        to_planes: &impl Fn(&Game::Position) -> Vec<Game::Bitboard>,
    ) -> (Vec<(Game::Move, f32)>, f32) {
        let planes = to_planes(pos);

        let batch_deadline = Duration::from_secs_f64(self.metrics.lock().unwrap().run_duration.get());
        let batch_deadline = batch_deadline.min(Duration::from_millis(20));
        let (move_scores, val) = self.batcher.apply(planes, batch_deadline, |inputs| {
            self.run_net(planes_to_tensor::<Game>(&inputs, self.batcher.batch_size()))
        });

        let moves = pos.legal_moves().collect_vec();
        let moves_probs = calc_moves_probs::<Game>(moves, &move_scores);
        (moves_probs, val)
    }
}

pub fn calc_moves_probs<Game: crate::game::Game>(
    moves: Vec<Game::Move>,
    move_scores: &[f32],
) -> Vec<(Game::Move, f32)> {
    let moves_scores = moves.iter().map(|m| move_scores[m.to_nn_idx()]).collect_vec();

    // Softmax normalization
    let max_p = moves_scores.iter().cloned().fold(f32::MIN, f32::max);
    let scores = moves_scores.into_iter().map(|p| (p - max_p).exp()).collect_vec();
    let p_sum: f32 = scores.iter().sum();
    let probs = scores.into_iter().map(|p| p / p_sum).collect_vec();

    moves.into_iter().zip(probs).collect_vec()
}

pub fn planes_to_tensor<Game: crate::game::Game>(samples: &[Vec<Game::Bitboard>], batch_size: usize) -> Array4<f32> {
    assert!(
        (1..=batch_size).contains(&samples.len()),
        "invalid sample len {}, 1..={}",
        samples.len(),
        batch_size
    );
    let planes_num = samples[0].len();
    let dims = (batch_size, planes_num, Game::BOARD_SIZE, Game::BOARD_SIZE);
    let mut tensor = Array4::<f32>::uninit(dims);

    for (b, sample) in samples.iter().enumerate() {
        for (c, plane) in sample.iter().enumerate() {
            for h in 0..(Game::BOARD_SIZE) {
                for w in 0..(Game::BOARD_SIZE) {
                    tensor[(b, c, h, w)].write(match plane.get(h * Game::BOARD_SIZE + w) {
                        true => 1.0,
                        false => 0.0,
                    });
                }
            }
        }
    }
    for i in samples.len()..batch_size {
        for c in 0..planes_num {
            for h in 0..(Game::BOARD_SIZE) {
                for w in 0..(Game::BOARD_SIZE) {
                    tensor[(i, c, h, w)].write(0.0);
                }
            }
        }
    }

    // Safety: we wrote to all elements
    unsafe { tensor.assume_init() }
}

pub fn flip_pos_if_needed<Position: crate::game::Position>(pos: Position) -> (Position, bool) {
    if pos.turn() == GameColor::Player1 {
        (pos, false)
    } else {
        (pos.flipped(), true)
    }
}

pub fn flip_score_if_needed<Move: crate::game::Move>(
    net_res: (Vec<(Move, f32)>, f32),
    pos_flipped: bool,
) -> (Vec<(Move, f32)>, f32) {
    if !pos_flipped {
        return net_res;
    }
    let (moves_probs, val) = net_res;

    /* Flip scalar value */
    let val = -val;

    /* Flip moves */
    let moves_probs = moves_probs.into_iter().map(|(m, p)| (m.flipped(), p)).collect_vec();

    (moves_probs, val)
}

struct Metrics {
    activation_count: metrics::Counter,
    run_duration: RunningAverage,
}
