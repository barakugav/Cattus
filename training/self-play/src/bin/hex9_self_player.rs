use cattus::hex::HexGame;
use cattus_self_play::self_play_cmd;

fn main() -> std::io::Result<()> {
    self_play_cmd::run_main::<HexGame<9>>()
}
