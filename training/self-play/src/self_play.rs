use itertools::Itertools;
use std::fs;
use std::ops::Deref;
use std::path::{self, Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;

use cattus::game::{GameColor, GameStatus, Move, Position};
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net;

use crate::serialize::DataSerializer;

pub struct DataEntry<Game: cattus::game::Game> {
    pub pos: Game::Position,
    pub probs: Vec<(Game::Move, f32)>,
    pub winner: Option<GameColor>,
}

impl<Game: cattus::game::Game> Clone for DataEntry<Game> {
    fn clone(&self) -> Self {
        DataEntry {
            pos: self.pos.clone(),
            probs: self.probs.clone(),
            winner: self.winner,
        }
    }
}

pub struct SerializerBase;
impl SerializerBase {
    pub fn write_entry<Game: cattus::game::Game>(
        planes: Vec<u64>,
        probs: Vec<(Game::Move, f32)>,
        winner: i8,
        filename: &Path,
    ) -> std::io::Result<()> {
        /* Use -1 for illegal moves */
        let mut probs_vec = vec![-1.0f32; Game::MOVES_NUM];

        /* Fill legal moves probabilities */
        for (m, prob) in probs {
            probs_vec[m.to_nn_idx()] = prob;
        }

        let u64bytes = u64::BITS as usize / 8;
        let f32bytes = /* f32::BITS */ 32 / 8;
        let i8bytes = i8::BITS as usize / 8;
        let size = planes.len() * u64bytes + probs_vec.len() * f32bytes + i8bytes;
        let mut bytes = Vec::with_capacity(size);

        /* Serialized in little indian format, should deserialized the same */
        bytes.extend(planes.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(probs_vec.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(winner.to_le_bytes());
        assert_eq!(bytes.len(), size);

        /* Write to file */
        fs::write(filename, bytes)
    }
}

#[derive(Copy, Clone)]
pub struct GamesResults {
    pub w1: u32,
    pub w2: u32,
    pub d: u32,
}

pub struct SelfPlayRunner<Game: cattus::game::Game> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    serializer: Arc<dyn DataSerializer<Game>>,
    thread_num: usize,
}

impl<Game: cattus::game::Game + 'static> SelfPlayRunner<Game> {
    pub fn new(
        player1_params: MctsParams<Game>,
        player2_params: MctsParams<Game>,
        serializer: Arc<dyn DataSerializer<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_params,
            player2_params,
            serializer,
            thread_num: thread_num as usize,
        }
    }

    pub fn generate_data(
        &self,
        games_num: usize,
        output_dir1: &Path,
        output_dir2: &Path,
    ) -> std::io::Result<GamesResults> {
        assert!(games_num % 2 == 0, "Games num should be a multiple of 2");

        /* Create output dir if doesn't exists */
        for output_dir in [output_dir1, output_dir2] {
            if !path::Path::new(output_dir).is_dir() {
                fs::create_dir_all(output_dir)?;
            }
        }

        let games_counter = Arc::new(AtomicUsize::new(0));
        let result = Arc::new(Mutex::new(GamesResults { w1: 0, w2: 0, d: 0 }));

        let job_builder = || {
            let worker = SelfPlayWorker::new(
                self.player1_params.clone(),
                self.player2_params.clone(),
                self.serializer.clone(),
                output_dir1.to_path_buf(),
                output_dir2.to_path_buf(),
                result.clone(),
                games_counter.clone(),
                games_num,
            );

            move || worker.generate_data().unwrap()
        };

        /* Spawn thread_num-1 to jobs [1..thread_num-1] */
        // TODO: add termination mechanism, detect if one of the threads panic
        let threads = (1..self.thread_num).map(|_| thread::spawn(job_builder())).collect_vec();

        /* Use current thread to do job 0 */
        job_builder()();

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        let res = *result.lock().unwrap().deref();
        Ok(res)
    }
}

struct SelfPlayWorker<Game: cattus::game::Game> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir1: PathBuf,
    output_dir2: PathBuf,
    results: Arc<Mutex<GamesResults>>,
    games_queue: Arc<AtomicUsize>,
    games_num: usize,
}

impl<Game: cattus::game::Game> SelfPlayWorker<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        player1_params: MctsParams<Game>,
        player2_params: MctsParams<Game>,
        serializer: Arc<dyn DataSerializer<Game>>,
        output_dir1: PathBuf,
        output_dir2: PathBuf,
        results: Arc<Mutex<GamesResults>>,
        games_queue: Arc<AtomicUsize>,
        games_num: usize,
    ) -> Self {
        Self {
            player1_params,
            player2_params,
            serializer,
            output_dir1,
            output_dir2,
            results,
            games_queue,
            games_num,
        }
    }

    fn generate_data(&self) -> std::io::Result<()> {
        let mut player1 = MctsPlayer::new(self.player1_params.clone());
        let mut player2 = MctsPlayer::new(self.player2_params.clone());

        loop {
            let game_idx = self.games_queue.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if game_idx >= self.games_num {
                break;
            }

            let mut game = Game::new();
            let mut pos_probs_pairs = Vec::new();
            let players_switch = game_idx % 2 == 1;

            let winner = loop {
                if let GameStatus::Finished(winner) = game.status() {
                    break winner;
                }

                let mut player = game.position().turn();
                if players_switch {
                    player = player.opposite()
                }
                let player = match player {
                    GameColor::Player1 => &mut player1,
                    GameColor::Player2 => &mut player2,
                };

                /* Generate probabilities from MCTS player */
                let moves = player.calc_moves_probabilities(game.pos_history());
                let next_move = player
                    .choose_move_from_probabilities(game.pos_history(), &moves)
                    .unwrap();

                /* Store probabilities */
                pos_probs_pairs.push((game.position().clone(), moves));

                /* Advance game position */
                game.play_single_turn(next_move);
            };

            /* Save all data entries */
            for (pos_idx, (pos, probs)) in pos_probs_pairs.into_iter().enumerate() {
                self.write_data_entry(game_idx, pos_idx, pos, probs, winner)?;
            }

            /* Update winning counters */
            {
                let mut results = self.results.lock().unwrap();
                let counter = match winner {
                    None => &mut results.d,
                    Some(mut player) => {
                        if players_switch {
                            player = player.opposite();
                        }
                        match player {
                            GameColor::Player1 => &mut results.w1,
                            GameColor::Player2 => &mut results.w2,
                        }
                    }
                };
                *counter += 1;
            }

            log::debug!("Game {} done", game_idx);
        }
        Ok(())
    }

    fn write_data_entry(
        &self,
        game_idx: usize,
        pos_idx: usize,
        pos: Game::Position,
        probs: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
    ) -> std::io::Result<()> {
        let output_dir = match pos.turn() {
            GameColor::Player1 => [&self.output_dir1, &self.output_dir2],
            GameColor::Player2 => [&self.output_dir2, &self.output_dir1],
        }[game_idx % 2];

        let winner = GameColor::to_signed_one(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (probs, winner) = net::flip_score_if_needed((probs, winner), is_flipped);
        let winner = match winner as i32 {
            1 => Some(GameColor::Player1),
            -1 => Some(GameColor::Player2),
            0 => None,
            other => panic!("unknown player index: {}", other),
        };

        self.serializer.serialize_data_entry(
            DataEntry { pos, probs, winner },
            &output_dir.join(format!("{game_idx:#08}_{pos_idx:#03}.traindata",)),
        )
    }
}
