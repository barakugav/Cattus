use crate::serialize::{generic_entry_to_bytes, DataEntry, ToBytes};
use cattus::game::{GameColor, Position};
use cattus::hex::HexGame;

impl<const BOARD_SIZE: usize> ToBytes for DataEntry<HexGame<BOARD_SIZE>> {
    fn to_bytes(&self) -> Vec<u8> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_signed_one(self.winner) as i8;
        assert_eq!(self.pos.turn(), GameColor::Player1);

        #[allow(clippy::identity_op)]
        let planes = cattus::hex::net::position_to_planes(&self.pos)
            .into_iter()
            .flat_map(|p| {
                [
                    ((p.get_raw() >> 00) & 0xffffffffffffffff) as u64,
                    ((p.get_raw() >> 64) & 0xffffffffffffffff) as u64,
                ]
                .into_iter()
            })
            .collect::<Vec<_>>();

        generic_entry_to_bytes::<HexGame<BOARD_SIZE>>(&planes, &self.probs, winner)
    }
}
