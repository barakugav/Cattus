use std::fmt::{self, Display};

use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};

pub fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("None"),
        Some(GameColor::Player1) => String::from("X"),
        Some(GameColor::Player2) => String::from("O"),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TttMove {
    idx: u8,
}

impl TttMove {
    pub fn new(r: usize, c: usize) -> Self {
        TttMove::from_idx(r * TttGame::BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        Self { idx: idx as u8 }
    }

    pub fn to_idx(&self) -> usize {
        self.idx as usize
    }

    pub fn row(&self) -> usize {
        self.idx as usize / TttGame::BOARD_SIZE
    }

    pub fn column(&self) -> usize {
        self.idx as usize % TttGame::BOARD_SIZE
    }
}

impl Move for TttMove {
    type Game = TttGame;

    fn flipped(&self) -> Self {
        *self
    }

    fn to_nn_idx(&self) -> usize {
        self.idx as usize
    }
}

impl Display for TttMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.column())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TttBitboard {
    bitmap: u16,
}

impl TttBitboard {
    pub fn get_raw(&self) -> u16 {
        self.bitmap
    }
}

impl Bitboard for TttBitboard {
    type Game = TttGame;

    fn new() -> Self {
        Self { bitmap: 0 }
    }

    fn full(val: bool) -> Self {
        Self {
            bitmap: if val { (1u16 << 9) - 1 } else { 0 },
        }
    }

    fn get(&self, idx: usize) -> bool {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        (self.bitmap & (1u16 << idx)) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        if val {
            self.bitmap |= 1u16 << idx;
        } else {
            self.bitmap &= !(1u16 << idx);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TttPosition {
    pub board_x: TttBitboard,
    pub board_o: TttBitboard,
    pub turn: GameColor,
    pub winner: Option<GameColor>,
}

impl TttPosition {
    /* Could lead to invalid board */
    pub fn from_bitboards(board_x: TttBitboard, board_o: TttBitboard, turn: GameColor) -> Self {
        let mut s = Self {
            board_x,
            board_o,
            turn,
            winner: None,
        };
        s.check_winner();
        s
    }

    pub fn pieces_x(&self) -> TttBitboard {
        self.board_x
    }

    pub fn pieces_o(&self) -> TttBitboard {
        self.board_o
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < TttGame::BOARD_SIZE && c < TttGame::BOARD_SIZE);
        let idx = r * TttGame::BOARD_SIZE + c;
        if self.board_x.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_o.get(idx) {
            return Some(GameColor::Player2);
        }
        None
    }

    pub fn make_move(&mut self, m: TttMove) {
        assert!(self.is_valid_move(m));

        match self.turn {
            GameColor::Player1 => &mut self.board_x,
            GameColor::Player2 => &mut self.board_o,
        }
        .set(m.to_idx(), true);

        self.turn = self.turn.opposite();
        self.check_winner();
    }

    pub fn is_valid_move(&self, m: TttMove) -> bool {
        if self.status().is_finished() {
            return false;
        }
        let idx = m.to_idx();
        !self.board_x.get(idx) && !self.board_o.get(idx)
    }

    pub fn check_winner(&mut self) {
        let winning_sequences = vec![
            0b111000000, // row 1
            0b000111000, // row 2
            0b000000111, // row 3
            0b100100100, // col 1
            0b010010010, // col 2
            0b001001001, // col 3
            0b100010001, // dial 1
            0b001010100, // dial 2
        ];

        for winning_sequence in winning_sequences {
            if (self.board_x.get_raw() & winning_sequence) == winning_sequence {
                self.winner = Some(GameColor::Player1);
                return;
            }
            if (self.board_o.get_raw() & winning_sequence) == winning_sequence {
                self.winner = Some(GameColor::Player2);
                return;
            }
        }
        self.winner = None;
    }
}

impl Display for TttPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..TttGame::BOARD_SIZE {
            for c in 0..TttGame::BOARD_SIZE {
                let ch = match self.get_tile(r, c) {
                    None => '_',
                    Some(GameColor::Player1) => 'X',
                    Some(GameColor::Player2) => 'O',
                };
                write!(f, "{} ", ch)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Position for TttPosition {
    type Game = TttGame;

    fn new() -> Self {
        TttPosition {
            board_x: Bitboard::new(),
            board_o: Bitboard::new(),
            turn: GameColor::Player1,
            winner: None,
        }
    }

    fn turn(&self) -> GameColor {
        self.turn
    }

    fn legal_moves(&self) -> impl Iterator<Item = TttMove> {
        (0..TttGame::BOARD_SIZE)
            .flat_map(|r| (0..TttGame::BOARD_SIZE).map(move |c| (r, c)))
            .filter_map(|(r, c)| {
                if self.get_tile(r, c).is_none() {
                    Some(TttMove::new(r, c))
                } else {
                    None
                }
            })
    }

    fn moved_position(&self, m: TttMove) -> Self {
        let mut res = *self;
        res.make_move(m);
        res
    }

    fn status(&self) -> GameStatus {
        if let Some(winner) = self.winner {
            return GameStatus::Finished(Some(winner));
        }
        if (self.board_x.get_raw() | self.board_o.get_raw()) == ((1 << 9) - 1) {
            return GameStatus::Finished(None);
        }
        GameStatus::Ongoing
    }

    fn flipped(&self) -> Self {
        Self {
            board_x: self.board_o,
            board_o: self.board_x,
            turn: self.turn.opposite(),
            winner: self.winner.map(|w| w.opposite()),
        }
    }
}

pub struct TttGame {
    pos_history: Vec<TttPosition>,
}
impl Game for TttGame {
    type Position = TttPosition;
    type Move = TttMove;
    type Bitboard = TttBitboard;
    const BOARD_SIZE: usize = 3;
    const MOVES_NUM: usize = Self::BOARD_SIZE * Self::BOARD_SIZE;
    const REPETITION_LIMIT: Option<usize> = None;

    fn new() -> Self {
        Self::from_position(TttPosition::new())
    }

    fn from_position(pos: Self::Position) -> Self {
        Self { pos_history: vec![pos] }
    }

    fn pos_history(&self) -> &[Self::Position] {
        &self.pos_history
    }

    fn status(&self) -> GameStatus {
        self.position().status()
    }

    fn play_single_turn(&mut self, next_move: Self::Move) {
        self.pos_history.push(self.position().moved_position(next_move));
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    use crate::game::player::{GamePlayer, PlayerRand};
    use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};
    use crate::ttt::{TttGame, TttMove, TttPosition};

    #[test]
    fn simple_game_and_mate() {
        let to_pos = |s: &str| ttt_position_from_str(s);
        assert_eq!(
            to_pos("xxxoo____o").status(),
            GameStatus::Finished(Some(GameColor::Player1))
        );
        assert_eq!(
            to_pos("oo_xxx___o").status(),
            GameStatus::Finished(Some(GameColor::Player1))
        );
        assert_eq!(
            to_pos("oo____xxxo").status(),
            GameStatus::Finished(Some(GameColor::Player1))
        );
        assert_eq!(
            to_pos("oxxo__ox_x").status(),
            GameStatus::Finished(Some(GameColor::Player2))
        );
        assert_eq!(
            to_pos("xox_o_xo_x").status(),
            GameStatus::Finished(Some(GameColor::Player2))
        );
        assert_eq!(
            to_pos("xxo__o_xox").status(),
            GameStatus::Finished(Some(GameColor::Player2))
        );
        assert_eq!(to_pos("xxoooxxxoo").status(), GameStatus::Finished(None));
    }

    #[test]
    fn flip() {
        for pos in vec![
            "oxx_o_o__o",
            "o_____xx_o",
            "xx_xx_xo_o",
            "ox___x_xox",
            "_x_o__o_xo",
            "ox__o____x",
            "_o__o_oxxo",
            "__xx_x__ox",
        ]
        .into_iter()
        .map(ttt_position_from_str)
        {
            assert_eq!(pos.turn().opposite(), pos.flipped().turn());
            assert_eq!(pos.flipped().flipped(), pos);
        }
    }

    #[test]
    fn flip_rand() {
        let mut rand = StdRng::seed_from_u64(0xf78ad6eae2a9ed60);

        let games_num = 100;
        for _ in 0..games_num {
            let mut player = PlayerRand::from_seed(rand.next_u64() ^ 0x9252a9837a0ddaca);
            let mut game = TttGame::new();

            while game.status().is_ongoing() {
                let pos = *game.position();
                let pos_t = pos.flipped();

                /* Assert flip of flip is original */
                assert_eq!(pos, pos_t.flipped());

                /* Assert flip of moves of flip are original moves */
                let moves: HashSet<TttMove> = HashSet::from_iter(pos.legal_moves());
                let moves_tt: HashSet<TttMove> = HashSet::from_iter(pos_t.legal_moves().map(|m| m.flipped()));
                assert_eq!(moves, moves_tt);

                /* Assert game result is the same */
                match (pos.status(), pos_t.status()) {
                    (GameStatus::Finished(c1), GameStatus::Finished(c2)) => assert_eq!(c1, c2.map(|c| c.opposite())),
                    (GameStatus::Ongoing, GameStatus::Ongoing) => {}
                    _ => panic!("One game ended but not the other"),
                }

                let next_move = <_ as GamePlayer<TttGame>>::next_move(&mut player, game.pos_history()).unwrap();
                game.play_single_turn(next_move);
            }
        }
    }

    pub fn ttt_position_from_str(s: &str) -> TttPosition {
        assert_eq!(
            s.chars().count(),
            TttGame::BOARD_SIZE * TttGame::BOARD_SIZE + 1,
            "unexpected string length"
        );
        let mut pos = TttPosition::new();
        for (idx, c) in s.chars().enumerate() {
            match idx.cmp(&(TttGame::BOARD_SIZE * TttGame::BOARD_SIZE)) {
                Ordering::Less => match c {
                    'x' => pos.board_x.set(idx, true),
                    'o' => pos.board_o.set(idx, true),
                    '_' => {}
                    _ => panic!("unknown board char: {:?}", c),
                },
                Ordering::Equal => {
                    pos.turn = match c {
                        'x' => GameColor::Player1,
                        'o' => GameColor::Player2,
                        _ => panic!("unknown turn char: {:?}", c),
                    }
                }
                Ordering::Greater => panic!("too many turn chars: {:?}", c),
            }
        }
        pos.check_winner();
        pos
    }
}
