pub mod cache;
pub mod value_func;

use itertools::Itertools;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand_distr::multi::{Dirichlet, MultiDistribution};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::Instant;

use crate::game::player::GamePlayer;
use crate::game::{GameColor, GameStatus, Position};
use crate::mcts::value_func::ValueFunction;
use crate::util::metric::RunningAverage;

/// Monte Carlo Tree Search (MCTS) implementation

#[derive(Clone, Copy)]
struct MctsNode<Position: crate::game::Position> {
    position: Position,
}

impl<Position: crate::game::Position> MctsNode<Position> {
    pub fn from_position(position: Position) -> Self {
        Self { position }
    }
}

#[derive(Clone, Copy)]
struct MctsEdge<Move> {
    m: Move,

    /// The initial score calculated for this node.
    /// In range [0, 1], "probability"
    init_score: f32,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    score_w: f32,
}

impl<Move> MctsEdge<Move> {
    pub fn new(m: Move, prior: f32, score: f32) -> Self {
        Self {
            m,
            init_score: prior,
            simulations_n: 0,
            score_w: score,
        }
    }
}

pub struct MctsPlayer<Game: crate::game::Game> {
    search_tree: DiGraph<MctsNode<Game::Position>, MctsEdge<Game::Move>>,
    root: Option<NodeIndex>,

    sim_num: u32,
    explore_factor: f32,
    temperature: TemperaturePolicy,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    value_func: Arc<dyn ValueFunction<Game>>,

    rng: StdRng,
    search_duration_metric: RunningAverage,
}

pub struct MctsParams<Game: crate::game::Game> {
    pub sim_num: u32,
    pub explore_factor: f32,
    pub temperature: TemperaturePolicy,
    pub prior_noise_alpha: f32,
    pub prior_noise_epsilon: f32,
    pub value_func: Arc<dyn ValueFunction<Game>>,
    pub seed: Option<u64>,
}
impl<Game: crate::game::Game> MctsParams<Game> {
    pub fn new(sim_num: u32, value_func: Arc<dyn ValueFunction<Game>>) -> Self {
        Self {
            sim_num,
            explore_factor: std::f32::consts::SQRT_2,
            temperature: TemperaturePolicy::constant(1.0),
            prior_noise_alpha: 0.0,
            prior_noise_epsilon: 0.0,
            value_func,
            seed: None,
        }
    }
}
impl<Game: crate::game::Game> Clone for MctsParams<Game> {
    fn clone(&self) -> Self {
        Self {
            sim_num: self.sim_num,
            explore_factor: self.explore_factor,
            temperature: self.temperature.clone(),
            prior_noise_alpha: self.prior_noise_alpha,
            prior_noise_epsilon: self.prior_noise_epsilon,
            value_func: Arc::clone(&self.value_func),
            seed: self.seed,
        }
    }
}

impl<Game: crate::game::Game> MctsPlayer<Game> {
    pub fn new(params: MctsParams<Game>) -> Self {
        assert!(params.sim_num > 0);
        assert!(params.explore_factor >= 0.0);
        assert!(params.prior_noise_alpha >= 0.0);
        assert!((0.0..=1.0).contains(&params.prior_noise_epsilon));

        let search_duration_metric_name = "mcts.search_duration";
        metrics::describe_gauge!(
            search_duration_metric_name,
            metrics::Unit::Seconds,
            "Duration of MCTS search"
        );
        let search_duration_metric = RunningAverage::new(0.99, metrics::gauge!(search_duration_metric_name));

        let rng = StdRng::seed_from_u64(params.seed.unwrap_or_else(rand::random));

        Self {
            search_tree: DiGraph::new(),
            root: None,

            sim_num: params.sim_num,
            explore_factor: params.explore_factor,
            prior_noise_alpha: params.prior_noise_alpha,
            prior_noise_epsilon: params.prior_noise_epsilon,
            temperature: params.temperature,
            value_func: params.value_func,

            rng,
            search_duration_metric,
        }
    }

    fn detect_repetition(&self, pos_history: &[Game::Position], trajectory: &[EdgeIndex]) -> bool {
        let repetition_limit = match Game::REPETITION_LIMIT {
            Some(l) if l > 1 => l,
            _ => return false,
        };

        let trajectory = trajectory.iter().map(|idx| {
            let (_e_source, e_target) = self.search_tree.edge_endpoints(*idx).unwrap();
            &self.search_tree[e_target].position
        });
        let full_trajectory = pos_history.iter().chain(trajectory);

        let mut repetitions = HashMap::new();
        for pos in full_trajectory {
            let repeat = repetitions.entry(pos).or_insert(0);
            *repeat += 1;
            if *repeat >= repetition_limit {
                return true;
            }
        }
        false
    }

    fn develop_tree(&mut self, pos_history: &[Game::Position]) {
        assert!(self.sim_num > 1);
        for _ in 0..self.sim_num {
            /* Select a leaf node */
            let path_to_selection = self.select();

            let repetition_reached = self.detect_repetition(pos_history, &path_to_selection);
            let leaf_id: NodeIndex = if path_to_selection.is_empty() {
                self.root.unwrap()
            } else {
                let (_e_source, e_target) = self
                    .search_tree
                    .edge_endpoints(*path_to_selection.last().unwrap())
                    .unwrap();
                e_target
            };
            let leaf_pos = &self.search_tree[leaf_id].position;

            let eval = if repetition_reached {
                0.0
            } else if let GameStatus::Finished(winner) = leaf_pos.status() {
                GameColor::to_signed_one(winner) as f32
            } else {
                /* Run value function once to obtain "simulation" value and initial children scores (probabilities) */
                let (per_move_val, eval) = self.simulate(leaf_id);

                /* Expand leaf and assign initial scores */
                self.create_children(leaf_id, per_move_val);

                /* Add Dirichlet noise to root initial probabilities */
                if leaf_id == self.root.unwrap() {
                    self.add_dirichlet_noise(leaf_id);
                }

                eval
            };

            /* back propagate the position score to the parents */
            self.backpropagate(path_to_selection, eval);
        }
    }

    /* Return path to selected leaf node */
    fn select(&self) -> Vec<EdgeIndex> {
        let mut path: Vec<EdgeIndex> = vec![];

        let mut node_id = self.root.unwrap();
        loop {
            let node = &self.search_tree[node_id];

            /* Node is leaf, done */
            if node.position.status().is_finished() || self.search_tree.edges(node_id).next().is_none() {
                return path;
            }

            let node_simcount = 1 + self
                .search_tree
                .edges(node_id)
                .map(|edge| edge.weight().simulations_n)
                .sum::<u32>();

            /* Node is not a leaf, choose best child and continue in it's sub tree */
            let edge = self
                .search_tree
                .edges(node_id)
                .max_by(|e1, e2| {
                    let val1 = self.calc_selection_heuristic(e1.weight(), node_simcount);
                    let val2 = self.calc_selection_heuristic(e2.weight(), node_simcount);
                    val1.partial_cmp(&val2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            path.push(edge.id());
            node_id = edge.target();
        }
    }

    fn calc_selection_heuristic(&self, edge: &MctsEdge<Game::Move>, parent_simcount: u32) -> f32 {
        let exploit = edge.score_w / edge.simulations_n.max(1) as f32;

        let explore =
            self.explore_factor * edge.init_score * ((parent_simcount as f32).sqrt() / (1 + edge.simulations_n) as f32);

        exploit + explore
    }

    fn create_children(&mut self, parent_id: NodeIndex, per_move_init_score: Vec<(Game::Move, f32)>) {
        let parent_pos = self.search_tree[parent_id].position.clone();
        debug_assert!(parent_pos.status().is_ongoing());

        debug_assert_eq!(
            {
                let moves_expected: HashSet<Game::Move> = HashSet::from_iter(parent_pos.legal_moves());
                moves_expected
            },
            {
                let moves_actual: HashSet<Game::Move> =
                    HashSet::from_iter(per_move_init_score.iter().map(|(m, _p)| m.clone()));
                moves_actual
            }
        );

        for (m, prior) in per_move_init_score {
            let leaf_pos = parent_pos.moved_position(m.clone());
            let (score, prior) = match leaf_pos.status() {
                GameStatus::Ongoing => (0.0, prior),
                GameStatus::Finished(game_color) => {
                    let score = GameColor::to_signed_one(game_color) as f32;
                    let score = match parent_pos.turn() {
                        GameColor::Player1 => score,
                        GameColor::Player2 => -score,
                    };
                    (score, 0.0)
                }
            };
            let leaf_id = self.search_tree.add_node(MctsNode::from_position(leaf_pos));
            self.search_tree
                .add_edge(parent_id, leaf_id, MctsEdge::new(m, prior, score));
        }
    }

    fn simulate(&self, leaf_id: NodeIndex) -> (Vec<(Game::Move, f32)>, f32) {
        let position = &self.search_tree[leaf_id].position;
        debug_assert!(position.status().is_ongoing());
        self.value_func.evaluate(position)
    }

    fn backpropagate(&mut self, path: Vec<EdgeIndex>, score: f32) {
        for edge_id in path {
            let (e_source, _e_target) = self.search_tree.edge_endpoints(edge_id).unwrap();
            let player_to_play = self.search_tree[e_source].position.turn();
            let edge = self.search_tree.edge_weight_mut(edge_id).unwrap();
            edge.simulations_n += 1;
            edge.score_w += match player_to_play {
                GameColor::Player1 => score,
                GameColor::Player2 => -score,
            };
        }
    }

    fn find_node_with_position(&self, position: &Game::Position, depth_limit: u32) -> Option<NodeIndex> {
        let mut layer = vec![self.root.unwrap()];

        for _ in 0..depth_limit {
            let mut next_layer = Vec::new();

            for node in layer {
                if &self.search_tree[node].position == position {
                    return Some(node);
                }
                // Add children to next layer
                for edge in self.search_tree.edges(node) {
                    next_layer.push(edge.target())
                }
            }
            layer = next_layer;
        }
        None
    }

    fn remove_all_but_subtree(&mut self, sub_tree_root: NodeIndex) {
        if self.root.unwrap() == sub_tree_root {
            return;
        }

        // In petgraph, when you remove a node, the indices of the other nodes
        // change. So instead of removing nodes from the current tree, we copy
        // the sub tree to a new graph
        let mut new_tree = DiGraph::new();
        let new_root = new_tree.add_node(self.search_tree[sub_tree_root].clone());
        let mut nodes = vec![(sub_tree_root, new_root)];

        while let Some((parent_old, parent_new)) = nodes.pop() {
            for edge in self.search_tree.edges(parent_old) {
                let child_old = edge.target();
                let child_data = &self.search_tree[child_old];
                let child_new = new_tree.add_node(child_data.clone());
                new_tree.add_edge(parent_new, child_new, edge.weight().clone());

                nodes.push((child_old, child_new));
            }
        }

        self.search_tree = new_tree;
        self.root = Some(new_root);

        /* If the prior probabilities of the new root were already calculated, add a Dirichlet noise */
        if self.search_tree.edges(new_root).next().is_some() {
            self.add_dirichlet_noise(new_root);
        }
    }

    pub fn calc_moves_probabilities(&mut self, pos_history: &[Game::Position]) -> Vec<(Game::Move, f32)> {
        let search_start_time = Instant::now();
        let position = pos_history.last().unwrap();

        if self.root.is_some() {
            // Tree was saved from the last search
            // Look for the position in the first three layers of the tree
            // TODO consider increasing depth limit
            match self.find_node_with_position(position, 3) {
                Some(node) => {
                    self.remove_all_but_subtree(node);
                }
                None => {
                    self.search_tree.clear();
                    self.root = None;
                }
            }
        }

        if self.root.is_none() {
            // Init search tree with one root node
            let root = MctsNode::from_position(position.clone());
            self.root = Some(self.search_tree.add_node(root));
        }
        assert_eq!(position, &self.search_tree[self.root.unwrap()].position);

        // Run all simulations
        self.develop_tree(pos_history);

        // create moves vector (move, sim_count)
        let moves_and_simcounts = self
            .search_tree
            .edges(self.root.unwrap())
            .map(|edge| {
                let e = edge.weight();
                (e.m.clone(), e.simulations_n)
            })
            .collect_vec();

        // normalize sim counts to create a valid distribution -> (move, simcount / simcount_total)
        let simcount_total: u32 = moves_and_simcounts.iter().map(|&(_, simcount)| simcount).sum();
        let res = moves_and_simcounts
            .into_iter()
            .map(|(m, simcount)| (m, simcount as f32 / simcount_total as f32))
            .collect_vec();

        self.search_duration_metric
            .set(search_start_time.elapsed().as_secs_f64());

        res
    }

    pub fn choose_move_from_probabilities(
        &mut self,
        pos_history: &[Game::Position],
        moves_probs: &[(Game::Move, f32)],
    ) -> Option<Game::Move> {
        if moves_probs.is_empty() {
            return None;
        }

        let temperature = self.temperature.get_temperature(pos_history.len() / 2);
        if temperature == 0.0 {
            let (m, _p) = moves_probs
                .iter()
                .max_by(|(_m1, p1), (_m2, p2)| p1.total_cmp(p2))
                .unwrap();
            Some(m.clone())
        } else {
            /* prob -> prob^temperature */
            assert!(temperature > 0.0);
            let probabilities = moves_probs
                .iter()
                .map(|(_m, p)| p.powf(1.0 / temperature))
                .collect_vec();

            /* normalize, prob -> prob / (probs sum) */
            let probs_sum: f32 = probabilities.iter().sum();
            let probabilities = probabilities.iter().map(|p| p / probs_sum).collect_vec();
            let distribution = WeightedIndex::new(probabilities).unwrap();
            Some(moves_probs[distribution.sample(&mut self.rng)].0.clone())
        }
    }

    fn add_dirichlet_noise(&mut self, node_id: NodeIndex) {
        if self.prior_noise_alpha == 0.0 || self.prior_noise_epsilon == 0.0 {
            return;
        }

        assert!((0.0..=1.0).contains(&self.prior_noise_epsilon));

        let moves = self.search_tree.edges(node_id).map(|e| e.id()).collect_vec();
        if moves.len() < 2 {
            return;
        }

        /* The Dirichlet implementation seems to return NaNs and INFs sometimes. */
        /* Keep drawing random noises until valid values are achieved */
        let dist = Dirichlet::new(&vec![self.prior_noise_alpha; moves.len()]).unwrap();
        let mut noise_vec = vec![0.0_f32; dist.sample_len()];
        loop {
            dist.sample_to_slice(&mut self.rng, &mut noise_vec);
            if noise_vec.iter().all(|n| n.is_finite()) {
                break;
            }
        }

        for (edge_id, noise) in moves.into_iter().zip(noise_vec) {
            let m = self.search_tree.edge_weight_mut(edge_id).unwrap();
            m.init_score = (1.0 - self.prior_noise_epsilon) * m.init_score + self.prior_noise_epsilon * noise;
            debug_assert!(m.init_score.is_finite());
        }
    }
}

impl<Game: crate::game::Game> GamePlayer<Game> for MctsPlayer<Game> {
    fn next_move(&mut self, pos_history: &[Game::Position]) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(pos_history);
        self.choose_move_from_probabilities(pos_history, &moves)
    }
}

#[derive(Clone)]
pub struct TemperaturePolicy {
    temperatures: Vec<(usize, f32)>,
    last_temperature: f32,
}

impl TemperaturePolicy {
    pub fn constant(temperature: f32) -> Self {
        assert!(temperature >= 0.0);
        Self {
            temperatures: Vec::new(),
            last_temperature: temperature,
        }
    }

    pub fn scheduled(temperatures: Vec<(usize, f32)>, last_temperature: f32) -> Self {
        assert!(temperatures.iter().all(|(_n, t)| *t >= 0.0));
        assert!(last_temperature >= 0.0);
        assert!(temperatures.windows(2).all(|w| w[0].0 < w[1].0)); // strictly increasing

        Self {
            temperatures,
            last_temperature,
        }
    }

    pub fn get_temperature(&self, move_num: usize) -> f32 {
        self.temperatures
            .iter()
            .find(|(threshold, _t)| move_num < *threshold)
            .map(|(_n, t)| *t)
            .unwrap_or(self.last_temperature)
    }
}
