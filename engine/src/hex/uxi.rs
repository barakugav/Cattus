use crate::game::player::GamePlayer;
use crate::game::{Bitboard, Game, GameColor, Position};
use crate::hex::{HexBitboard, HexGameStandard, HexMove, HexPosition};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::{io, process, thread, time};

/**
 * UXI (Universal Hex Interface), similar to UCI (Universal Chess Interface) is a general
 * interface for Hex engine communication. All communication is done by standard input and
 * output, each command has '\n' at the end.
 *
 * Input commands (from host to engine):
 *      next_move [pos] [color]
 *          calculate the next move of the engine.
 *              [pos] - current position, string of 121 characters, e/r/b (empty, red, blue),
 *                  i'th character corresponding to the [i/11][i%11] tile
 *              [color] - the engine color, one character, r/b (red, blue)
 *      quit
 *          quit from the program, the engine should exit in 0.1 sec
 * Output commands (from engine to host):
 *      ready
 *          after engine was started, this is the first command it should issue to let the host know its ready
 *      move [indices]
 *          the next move of the engine
 *              indices - the engine move, two numbers with comma separation, "c,r"
 */
pub struct HexPlayerUXI {
    exe_filename: PathBuf,
    // err_filename: PathBuf,
    // err_file: Option<File>,
    process: Option<process::Child>,
}

impl HexPlayerUXI {
    pub fn new(exe_filename: PathBuf /* , err_filename: PathBuf */) -> Self {
        Self {
            exe_filename,
            // err_filename,
            // err_file: None,
            process: None,
        }
    }

    pub fn start(&mut self, engine_params: &[String]) -> bool {
        if self.process.is_some() {
            println!("[UXIPlayer] Process is already launched");
            return false;
        }

        println!(
            "[UXIPlayer] launching engine: {} {:?}",
            self.exe_filename.display(),
            engine_params
        );

        // assert!(self.err_file.is_none());
        // match File::create(self.err_filename) {
        //     Err(error) => {
        //         eprintln!("Failed to open error file for process: {}", error);
        //         self.err_file = None;
        //         return false;
        //     }
        //     Ok(file) => self.err_file = file,
        // }

        self.process = match process::Command::new(self.exe_filename.clone())
            .args(engine_params)
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            // .stderr(process::Stdio::from_raw_fd(
            //     self.err_file.unwrap().into_raw_fd(),
            // ))
            .spawn()
        {
            Err(error) => {
                eprintln!("[UXIPlayer] Failed to launch process: {error}");
                None
            }
            Ok(process) => Some(process),
        };
        if self.process.is_none() {
            return false;
        }

        let Some(resp) = self.receive_command() else {
            return false;
        };
        let response = resp.split(' ').collect::<Vec<_>>();
        if response.is_empty() {
            return false;
        }
        match response[0] {
            "ready" => true,
            _ => {
                eprintln!("[UXIPlayer] Unexpected command: {response:?} (expected ready)");
                false
            }
        }
    }

    pub fn stop(&mut self) {
        if self.process.is_some() {
            /* be nice (for 0.1 sec) */
            self.send_command("quit");
            thread::sleep(time::Duration::from_millis(100));

            let mut kill_needed = false;
            match self.process.as_mut().unwrap().try_wait() {
                Err(error) => {
                    eprintln!("[UXIPlayer] Failed to get engine process status: {error}");
                    kill_needed = true;
                }
                Ok(status) => match status {
                    None => {
                        eprintln!("[UXIPlayer] Engine process ignored 'quit' command");
                        kill_needed = true;
                    }
                    Some(_) => { /* engine quit by it's own */ }
                },
            };
            if kill_needed {
                /* don't be nice */
                if let Err(error) = self.process.as_mut().unwrap().kill() {
                    eprintln!("[UXIPlayer] Failed to kill process: {error}");
                }
            }
            self.process = None;
        }
    }

    fn send_command(&mut self, cmd: &str) {
        if self.process.is_none() {
            eprintln!("[UXIPlayer] Engine was not started.");
            return;
        }
        let process = self.process.as_mut().unwrap();
        let engine_stdin = process.stdin.as_mut().unwrap();
        if let Err(error) = engine_stdin.write(format!("{}\n", cmd.trim()).as_bytes()) {
            eprintln!("[UXIPlayer] Failed to pass command: {error}")
        }
    }

    fn receive_command(&mut self) -> Option<String> {
        if self.process.is_none() {
            eprintln!("[UXIPlayer] Engine was not started.");
            return None;
        }
        let process = self.process.as_mut().unwrap();
        let mut engine_stdout = BufReader::new(process.stdout.as_mut().unwrap());
        let mut output_line = String::new();

        match engine_stdout.read_line(&mut output_line) {
            Err(error) => {
                eprintln!("[UXIPlayer] Failed to read output from engine: {error}");
                None
            }
            Ok(_) => Some(String::from(output_line.trim())),
        }
    }
}

impl GamePlayer<HexGameStandard> for HexPlayerUXI {
    fn next_move(
        &mut self,
        pos_history: &[<HexGameStandard as Game>::Position],
    ) -> Option<<HexGameStandard as Game>::Move> {
        let mut command = String::with_capacity(10 + HexGameStandard::BOARD_SIZE * HexGameStandard::BOARD_SIZE + 3);
        command.push_str("next_move ");
        position_to_uxi(pos_history.last().unwrap(), &mut command);
        self.send_command(&command);
        let resp = self.receive_command()?;
        let response = resp.split(' ').collect::<Vec<_>>();
        if response.is_empty() {
            return None;
        }
        match response[0] {
            "move" => {
                if response.len() != 2 {
                    eprintln!("[UXIPlayer] Expected \"move r,c\" format: \"{resp}\"");
                    return None;
                }
                let m_str = response[1].split(',').collect::<Vec<_>>();
                if m_str.len() != 2 {
                    eprintln!("[UXIPlayer] Expected \"move r,c\" format: \"{resp}\"");
                    return None;
                }
                let r = match m_str[0].parse::<usize>() {
                    Err(error) => {
                        eprintln!("[UXIPlayer] Failed to parse row index: {error}");
                        return None;
                    }
                    Ok(row) => row,
                };
                let c = match m_str[1].parse::<usize>() {
                    Err(error) => {
                        eprintln!("[UXIPlayer] Failed to parse column index: {error}");
                        return None;
                    }
                    Ok(column) => column,
                };
                Some(HexMove::new(r, c))
            }
            unknown_cmd => {
                eprintln!("[UXIPlayer] Unknown command: {unknown_cmd}");
                None
            }
        }
    }
}

pub struct UxiEngine {
    player: Box<dyn GamePlayer<HexGameStandard>>,
}

impl UxiEngine {
    pub fn new(player: Box<dyn GamePlayer<HexGameStandard>>) -> Self {
        Self { player }
    }

    pub fn run(&mut self) {
        println!("ready");
        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("[UxiEngine] failed to read input");
            let args = line.split_whitespace().collect::<Vec<_>>();

            if args.is_empty() {
                continue;
            }
            match args[0] {
                "next_move" => {
                    if args.len() != 3 {
                        eprintln!("[UxiEngine] Expected position and color for next_move command.");
                        continue;
                    }
                    let pos_str = args[1];
                    let color_str = args[2];
                    let Some(pos) = uxi_to_position(pos_str, color_str) else {
                        eprintln!("[UxiEngine] Failed to parse position.");
                        continue;
                    };
                    match self.player.next_move(&[pos]) {
                        None => println!("error"),
                        Some(m) => println!("move {},{}", m.row(), m.column()),
                    };
                }
                "quit" => {
                    break;
                }
                unknown_cmd => {
                    eprintln!("[UxiEngine] Unknown command: {unknown_cmd}");
                }
            }
        }
    }
}

fn position_to_uxi(position: &<HexGameStandard as Game>::Position, s: &mut String) {
    for r in 0..HexGameStandard::BOARD_SIZE {
        for c in 0..HexGameStandard::BOARD_SIZE {
            s.push(match position.get_tile(r, c) {
                None => 'e',
                Some(GameColor::Player1) => 'r',
                Some(GameColor::Player2) => 'b',
            });
        }
    }
    s.push(' ');
    s.push(match position.turn() {
        GameColor::Player1 => 'r',
        GameColor::Player2 => 'b',
    });
}

fn uxi_to_position(pos_str: &str, color_str: &str) -> Option<<HexGameStandard as Game>::Position> {
    let mut board_red = HexBitboard::new();
    let mut board_blue = HexBitboard::new();
    let mut idx = 0;
    for tile in pos_str.chars() {
        if idx >= HexGameStandard::BOARD_SIZE * HexGameStandard::BOARD_SIZE {
            eprintln!("Too many chars in position string");
            return None;
        }
        match tile {
            'e' => {}
            'r' => board_red.set(idx, true),
            'b' => board_blue.set(idx, true),
            unknown_tile => {
                eprintln!("Unknown tile: {unknown_tile}");
                return None;
            }
        };
        idx += 1;
    }
    if idx != HexGameStandard::BOARD_SIZE * HexGameStandard::BOARD_SIZE {
        eprintln!("Too few chars in position string");
        return None;
    }
    let player = match color_str {
        "r" => GameColor::Player1,
        "b" => GameColor::Player2,
        unknown_player => {
            eprintln!("Unknown player: {unknown_player}");
            return None;
        }
    };
    Some(HexPosition::new_from_board(board_red, board_blue, player))
}
