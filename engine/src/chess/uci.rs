use crate::chess::{ChessGame, ChessMove, ChessPosition};
use crate::game::player::GamePlayer;
use crate::game::Position;
use crate::mcts::{MctsParams, MctsPlayer};
use std::collections::HashMap;
use std::io;

pub struct UCI {
    player_params: MctsParams<ChessGame>,
    options: HashMap<String, String>,
    player: Option<MctsPlayer<ChessGame>>,
    pos_history: Option<Vec<ChessPosition>>,
    best_move: Option<ChessMove>,
}

impl UCI {
    pub fn new(player_params: MctsParams<ChessGame>) -> Self {
        Self {
            player_params,
            options: HashMap::new(),
            player: None,
            pos_history: None,
            best_move: None,
        }
    }

    pub fn run(&mut self) {
        let mut line = String::new();
        loop {
            line.clear();
            if let Err(e) = io::stdin().read_line(&mut line) {
                eprintln!("read line error: {e}");
                continue;
            }
            let line = line.trim();
            log(format!("[From GUI] {line}"));
            let (command, args) = Self::parse_command(line);

            match command {
                "uci" => {
                    self.send_response("id name _PROJECT_NAME_TODO_ v1.0.0");
                    self.send_response("id author Barak Ugav Yishai Gronich");
                    /* TODO send options */
                    self.send_response("uciok");
                }
                "isready" => self.send_response("readyok"),
                "setoption" => {
                    let args = Self::parse_args(&args, &["name", "value"]);
                    let name = args.value("name").expect("setoption requires 'name' arg");
                    let value = args.value("value").expect("setoption requires 'value' arg");
                    self.options.insert(name.to_string(), value.to_string());
                }
                "ucinewgame" => self.player = Some(MctsPlayer::new(self.player_params.clone())),
                "position" => self.cmd_position(&args),
                "go" => self.cmd_go(&args),
                "stop" => {}
                "ponderhit" => println!("uciok"),
                "start" => println!("uciok"),
                "fen" => println!("uciok"),
                "xyzzy" => println!("uciok"),
                "quit" => return,
                _ => {
                    eprintln!("unknown command {command}");
                    continue;
                }
            }
        }
    }

    pub fn cmd_position(&mut self, args: &[&str]) {
        let args = Self::parse_args(args, &["fen", "startpos", "moves"]);
        let fen = args.value("fen");
        let startpos = args.flag("startpos");
        let moves = args.values_iter("moves");

        assert_ne!(fen.is_some(), startpos, "position cmd requires either fen or startpos");
        let mut pos = fen.map(ChessPosition::from_fen).unwrap_or_else(ChessPosition::new);
        let mut pos_history = vec![pos];
        for move_str in moves {
            let m = ChessMove::from_lan(move_str).unwrap();
            pos = pos.moved_position(m);
            pos_history.push(pos);
        }
        self.pos_history = Some(pos_history);
    }

    pub fn cmd_go(&mut self, args: &[&str]) {
        let args = Self::parse_args(
            args,
            &[
                "searchmoves",
                "ponder",
                "wtime",
                "btime",
                "winc",
                "binc",
                "movestogo",
                "depth",
                "nodes",
                "mate",
                "movetime",
                "infinite",
            ],
        );

        #[allow(unused)]
        struct GoParams {
            searchmoves: Vec<String>,
            ponder: bool,
            wtime: Option<u64>,
            btime: Option<u64>,
            winc: Option<u64>,
            binc: Option<u64>,
            movestogo: Option<u32>,
            depth: Option<u32>,
            nodes: Option<u32>,
            movetime: Option<u64>,
            infinite: bool,
        }
        #[allow(unused)]
        let go_args = GoParams {
            searchmoves: args
                .values_iter("searchmoves")
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            ponder: args.flag("ponder"),
            wtime: args.value("wtime").and_then(|s| s.parse::<u64>().ok()),
            btime: args.value("btime").and_then(|s| s.parse::<u64>().ok()),
            winc: args.value("winc").and_then(|s| s.parse::<u64>().ok()),
            binc: args.value("binc").and_then(|s| s.parse::<u64>().ok()),
            movestogo: args.value("movestogo").and_then(|s| s.parse::<u32>().ok()),
            depth: args.value("depth").and_then(|s| s.parse::<u32>().ok()),
            nodes: args.value("nodes").and_then(|s| s.parse::<u32>().ok()),
            movetime: args.value("movetime").and_then(|s| s.parse::<u64>().ok()),
            infinite: args.flag("infinite"),
        };

        let player = self.player.as_mut().unwrap();
        self.best_move = Some(player.next_move(self.pos_history.as_ref().unwrap()).unwrap());
        self.send_response(format!("bestmove {}", self.best_move.unwrap()));
    }

    fn send_response<S: Into<String>>(&self, s: S) {
        let s = s.into();
        log(format!("[To GUI] {s}"));
        println!("{s}");
    }

    fn parse_command(s: &str) -> (&str, Vec<&str>) {
        let mut words = s
            .split(' ')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        assert!(!words.is_empty(), "empty command");
        let command = words.remove(0);
        (command, words)
    }

    fn parse_args<'a>(args: &[&'a str], keys: &[&str]) -> CommandArgs<'a> {
        let mut key = None;
        let mut map = HashMap::new();
        for arg in args {
            if keys.contains(arg) {
                let old_val = map.insert(*arg, Vec::new());
                assert!(old_val.is_none(), "arg '{arg}' appears multiple times");
                key = Some(*arg);
            } else {
                assert!(key.is_some(), "arg '{arg}' has no key");
                map.get_mut(key.unwrap()).unwrap().push(*arg);
            }
        }
        CommandArgs::new(map)
    }
}

struct CommandArgs<'a> {
    args: HashMap<&'a str, Vec<&'a str>>,
}
impl<'a> CommandArgs<'a> {
    fn new(args: HashMap<&'a str, Vec<&'a str>>) -> Self {
        Self { args }
    }

    fn value(&self, key: &str) -> Option<&'a str> {
        let values = self.args.get(key)?;
        assert_eq!(values.len(), 1, "arg '{key}' should have exactly one value");
        Some(values[0])
    }

    fn values(&self, key: &str) -> Option<&Vec<&'a str>> {
        self.args.get(key)
    }

    fn values_iter(&self, key: &str) -> impl Iterator<Item = &'a str> {
        self.values(key).into_iter().flatten().copied()
    }

    fn flag(&self, key: &str) -> bool {
        match self.args.get(key) {
            Some(values) => {
                assert!(values.is_empty(), "flag arg '{key}' should have no values");
                true
            }
            None => false,
        }
    }
}

fn log(_s: impl AsRef<str>) {
    // let mut file = std::fs::File::options()
    //     .write(true)
    //     .append(true)
    //     .create(true)
    //     .open("/Users/barak/code/Cattus/uci_log.txt")
    //     .unwrap();
    // <_ as std::io::Seek>::seek(&mut file, std::io::SeekFrom::End(0))
    //     .expect("failed to seek to end of log file");
    // <_ as std::io::Write>::write(&mut file, format!("{}\n", _s.as_ref()).as_bytes())
    //     .expect("failed to append to log file");
}
