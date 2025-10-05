use cattus::hex::{HexGame, HEX_STANDARD_BOARD_SIZE};
use cattus_self_play::self_play_cmd;

fn main() -> std::io::Result<()> {
    self_play_cmd::run_main::<HexGame<HEX_STANDARD_BOARD_SIZE>>()
}
