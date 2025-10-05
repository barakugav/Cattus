use std::path::Path;

use crate::self_play::{DataEntry, SerializerBase};
use crate::serialize::DataSerializer;
use cattus::game::{GameColor, Position};
use cattus::hex::HexGame;
use itertools::Itertools;

pub struct HexSerializer;
impl<const BOARD_SIZE: usize> DataSerializer<HexGame<BOARD_SIZE>> for HexSerializer {
    fn serialize_data_entry(&self, entry: DataEntry<HexGame<BOARD_SIZE>>, filename: &Path) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_signed_one(entry.winner) as i8;
        assert_eq!(entry.pos.turn(), GameColor::Player1);

        #[allow(clippy::identity_op)]
        let planes = cattus::hex::net::position_to_planes(&entry.pos)
            .into_iter()
            .flat_map(|p| {
                [
                    ((p.get_raw() >> 00) & 0xffffffffffffffff) as u64,
                    ((p.get_raw() >> 64) & 0xffffffffffffffff) as u64,
                ]
                .into_iter()
            })
            .collect_vec();

        SerializerBase::write_entry::<HexGame<BOARD_SIZE>>(planes, entry.probs, winner, filename)
    }
}
