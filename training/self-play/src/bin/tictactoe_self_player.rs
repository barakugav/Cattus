use cattus::ttt::TttGame;
use cattus_self_play::self_play_cmd::run_main;

fn main() -> std::io::Result<()> {
    run_main::<TttGame>()
}
