use crate::serialize::{generic_entry_to_bytes, DataEntry, ToBytes};
use cattus::game::{GameColor, Position};
use cattus::ttt::TttGame;

impl ToBytes for DataEntry<TttGame> {
    fn to_bytes(&self) -> Vec<u8> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_signed_one(self.winner) as i8;
        assert_eq!(self.pos.turn(), GameColor::Player1);

        let planes = cattus::ttt::net::position_to_planes(&self.pos)
            .iter()
            .map(|p| p.get_raw() as u64)
            .collect::<Vec<_>>();

        generic_entry_to_bytes::<TttGame>(&planes, &self.probs, winner)
    }
}
