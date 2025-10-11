use std::io::Write;
use std::path::{Path, PathBuf};

use cattus::chess::{ChessGame, ChessPosition};
use cattus::game::{GameColor, Position};
use cattus::hex::HexGame;
use cattus::ttt::TttGame;
use cattus_self_play::serialize::{DataEntry, ToBytes};
use cattus_self_play::test_util::{hex_position_from_str, ttt_position_from_str};
use clap::Parser;
use itertools::Itertools;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    game: String,
    #[clap(long)]
    position: String,
    #[clap(long)]
    outfile: PathBuf,
}

fn main() -> std::io::Result<()> {
    cattus::util::init_globals(Default::default());

    let args = Args::parse();
    match args.game.as_str() {
        "tictactoe" => test_tictactoe(args),
        "hex4" => test_hex::<4>(args),
        "hex5" => test_hex::<5>(args),
        "hex7" => test_hex::<7>(args),
        "hex9" => test_hex::<9>(args),
        "hex11" => test_hex::<11>(args),
        "chess" => test_chess(args),
        unknown_game => panic!("unknown game: {:?}", unknown_game),
    }
}

fn test_tictactoe(args: Args) -> std::io::Result<()> {
    let pos = ttt_position_from_str(&args.position);
    serialize_position::<TttGame>(pos, &args.outfile)
}

fn test_hex<const BOARD_SIZE: usize>(args: Args) -> std::io::Result<()> {
    let pos = hex_position_from_str(&args.position);
    serialize_position::<HexGame<BOARD_SIZE>>(pos, &args.outfile)
}

fn test_chess(args: Args) -> std::io::Result<()> {
    let pos = ChessPosition::from_fen(&args.position);
    serialize_position::<ChessGame>(pos, &args.outfile)
}

fn serialize_position<Game>(pos: Game::Position, filename: &Path) -> std::io::Result<()>
where
    Game: cattus::game::Game,
    DataEntry<Game>: ToBytes,
{
    let moves = pos.legal_moves().collect_vec();
    let moves_num = moves.len();
    let probs = moves
        .into_iter()
        .enumerate()
        .map(|(idx, m)| (m, idx as f32 / (moves_num * (moves_num - 1)) as f32))
        .collect_vec();
    let winner = match moves_num % 3 {
        0 => Some(GameColor::Player1),
        1 => Some(GameColor::Player2),
        2 => None,
        _ => panic!("cant happen"),
    };
    let bytes = DataEntry { pos, probs, winner }.to_bytes();
    std::fs::File::create_new(filename)?.write_all(&bytes)
}
