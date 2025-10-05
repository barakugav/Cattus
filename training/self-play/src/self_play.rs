use crossbeam::channel::{Receiver, Sender};
use std::fs;
use std::io::Write;
use std::path::{self, Path, PathBuf};

use cattus::game::{GameColor, GameStatus, Position};
use cattus::mcts::{MctsParams, MctsPlayer};
use cattus::net;

use crate::serialize::{DataEntry, ToBytes};
use crate::util::thread::{ThreadControl, ThreadManager};

#[derive(Copy, Clone)]
pub struct GamesResults {
    pub w1: u32,
    pub w2: u32,
    pub d: u32,
}

pub struct SelfPlayRunner<Game: cattus::game::Game> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    thread_num: usize,
}

impl<Game> SelfPlayRunner<Game>
where
    Game: cattus::game::Game + 'static,
    DataEntry<Game>: ToBytes,
{
    pub fn new(player1_params: MctsParams<Game>, player2_params: MctsParams<Game>, thread_num: u32) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_params,
            player2_params,
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

        let (task_sender, task_receiver) = crossbeam::channel::unbounded();
        let (results_sender, results_receiver) = crossbeam::channel::unbounded();

        let mut manager = ThreadManager::new();
        for i in 0..self.thread_num.max(1) {
            let worker = SelfPlayWorker {
                player1_params: self.player1_params.clone(),
                player2_params: self.player2_params.clone(),
                output_dir1: output_dir1.to_path_buf(),
                output_dir2: output_dir2.to_path_buf(),
                task_channel: task_receiver.clone(),
                results_channel: results_sender.clone(),
            };

            manager.spawn_thread(format!("self_play_worker_{}", i), move |control| {
                control.set_ready();

                worker.generate_data(control).unwrap();
            });
        }
        manager.wait_ready(std::time::Instant::now() + std::time::Duration::from_secs(10));

        for game_idx in 0..games_num {
            task_sender.send(SelfPlayTask { game_idx }).unwrap();
        }
        let mut results = Vec::new();
        while results.len() < games_num {
            let res = results_receiver.recv_timeout(std::time::Duration::from_secs(1));
            if manager.any_thread_crashed() {
                break;
            }
            if let Ok(res) = res {
                results.push(res);
            }
        }
        let join_res = manager.terminate();
        if let Err(e) = join_res {
            return Err(std::io::Error::other(format!("Thread panicked: {:?}", e)));
        }
        drop(task_sender);

        let mut summery = GamesResults { w1: 0, w2: 0, d: 0 };
        for res in results {
            match res {
                Some(GameColor::Player1) => summery.w1 += 1,
                Some(GameColor::Player2) => summery.w2 += 1,
                None => summery.d += 1,
            }
        }
        Ok(summery)
    }
}

struct SelfPlayTask {
    game_idx: usize,
}

struct SelfPlayWorker<Game: cattus::game::Game> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    output_dir1: PathBuf,
    output_dir2: PathBuf,
    task_channel: Receiver<SelfPlayTask>,
    results_channel: Sender<Option<GameColor>>,
}

impl<Game> SelfPlayWorker<Game>
where
    Game: cattus::game::Game,
    DataEntry<Game>: ToBytes,
{
    fn generate_data(&self, threads_control: ThreadControl) -> std::io::Result<()> {
        let mut player1 = MctsPlayer::new(self.player1_params.clone());
        let mut player2 = MctsPlayer::new(self.player2_params.clone());

        loop {
            let task = crossbeam::select! {
                recv(self.task_channel) -> task => {
                    match task {
                        Ok(task) => task,
                        Err(_) => break, // channel closed
                    }
                }
                recv(threads_control.termination_receiver()) -> _ => {
                    break;
                }
            };

            let mut game = Game::new();
            let mut pos_probs_pairs = Vec::new();
            let players_switch = task.game_idx % 2 == 1;

            let winner = loop {
                if threads_control.termination_receiver().try_recv().is_ok() {
                    return Ok(());
                }
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
                self.write_data_entry(task.game_idx, pos_idx, pos, probs, winner)?;
            }

            /* Update winning counters */
            self.results_channel
                .send(winner.map(|c| if players_switch { c.opposite() } else { c }))
                .unwrap();

            log::debug!("Game {} done", task.game_idx);
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

        let data_entry_bytes = DataEntry { pos, probs, winner }.to_bytes();
        std::fs::File::create_new(output_dir.join(format!("{game_idx:#08}_{pos_idx:#03}.traindata")))?
            .write_all(&data_entry_bytes)
    }
}
