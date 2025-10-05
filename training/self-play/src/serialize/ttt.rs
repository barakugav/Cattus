use std::path::Path;

use itertools::Itertools;

use crate::self_play::{DataEntry, SerializerBase};
use crate::serialize::DataSerializer;
use cattus::game::{GameColor, Position};
use cattus::ttt::TttGame;

pub struct TttSerializer;
impl DataSerializer<TttGame> for TttSerializer {
    fn serialize_data_entry(&self, entry: DataEntry<TttGame>, filename: &Path) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_signed_one(entry.winner) as i8;
        assert_eq!(entry.pos.turn(), GameColor::Player1);

        let planes = cattus::ttt::net::position_to_planes(&entry.pos)
            .iter()
            .map(|p| p.get_raw() as u64)
            .collect_vec();

        SerializerBase::write_entry::<TttGame>(planes, entry.probs, winner, filename)
    }
}
