use std::fmt::{self, Display};

use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexMove<const BOARD_SIZE: usize> {
    idx: u16,
}

impl<const BOARD_SIZE: usize> HexMove<BOARD_SIZE> {
    pub fn new(r: usize, c: usize) -> Self {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        HexMove::from_idx(r * BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        Self { idx: idx as u16 }
    }

    pub fn to_idx(&self) -> usize {
        self.idx as usize
    }

    pub fn row(&self) -> usize {
        self.idx as usize / BOARD_SIZE
    }

    pub fn column(&self) -> usize {
        self.idx as usize % BOARD_SIZE
    }
}

impl<const BOARD_SIZE: usize> Move for HexMove<BOARD_SIZE> {
    type Game = HexGame<BOARD_SIZE>;

    fn flipped(&self) -> Self {
        HexMove::new(self.column(), self.row())
    }

    fn to_nn_idx(&self) -> usize {
        self.idx as usize
    }
}

impl<const BOARD_SIZE: usize> Display for HexMove<BOARD_SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.column())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexBitboard<const BOARD_SIZE: usize> {
    bitmap: u128,
}

impl<const BOARD_SIZE: usize> HexBitboard<BOARD_SIZE> {
    pub fn get_raw(&self) -> u128 {
        self.bitmap
    }

    fn flip(&self) -> Self {
        let mut f = HexBitboard::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let idx = r * BOARD_SIZE + c;
                let idxf = c * BOARD_SIZE + r;
                f.set(idxf, self.get(idx));
            }
        }
        f
    }

    fn is_empty(&self) -> bool {
        self.bitmap == 0
    }
}

impl<const BOARD_SIZE: usize> Bitboard for HexBitboard<BOARD_SIZE> {
    type Game = HexGame<BOARD_SIZE>;
    fn new() -> Self {
        Self { bitmap: 0 }
    }

    fn full(val: bool) -> Self {
        assert!(BOARD_SIZE * BOARD_SIZE <= u128::BITS as usize);
        Self {
            bitmap: if val {
                (1u128 << (BOARD_SIZE * BOARD_SIZE)) - 1
            } else {
                0
            },
        }
    }

    fn get(&self, idx: usize) -> bool {
        assert!(BOARD_SIZE * BOARD_SIZE <= u128::BITS as usize);
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        (self.bitmap & (1u128 << idx)) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        assert!(BOARD_SIZE * BOARD_SIZE <= u128::BITS as usize);
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        if val {
            self.bitmap |= 1u128 << idx;
        } else {
            self.bitmap &= !(1u128 << idx);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexPosition<const BOARD_SIZE: usize> {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted left. So, board[0][0] is the "top left end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][BOARD_SIZE - 1] is the "bottom end".
    /// Red tries to connect top-bottom and blue tries to connect left-right.
    pub board_red: HexBitboard<BOARD_SIZE>,
    pub board_blue: HexBitboard<BOARD_SIZE>,
    pub turn: GameColor,

    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with red pieces */
    top_red_reach: HexBitboard<BOARD_SIZE>,
    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with blue pieces */
    left_blue_reach: HexBitboard<BOARD_SIZE>,
    number_of_empty_tiles: u16,
    winner: Option<GameColor>,
}

impl<const BOARD_SIZE: usize> HexPosition<BOARD_SIZE> {
    pub fn new_with_starting_color(starting_color: GameColor) -> Self {
        Self {
            board_red: HexBitboard::new(),
            board_blue: HexBitboard::new(),
            turn: starting_color,
            top_red_reach: HexBitboard::new(),
            left_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u16,
            winner: None,
        }
    }

    pub fn new_from_board(
        board_red: HexBitboard<BOARD_SIZE>,
        board_blue: HexBitboard<BOARD_SIZE>,
        turn: GameColor,
    ) -> Self {
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let idx = r * BOARD_SIZE + c;
                assert!(
                    !(board_red.get(idx) && board_blue.get(idx)),
                    "invalid board: both players have piece at ({r}, {c})"
                );
            }
        }

        let mut position = Self {
            board_red,
            board_blue,
            turn,
            top_red_reach: HexBitboard::new(),
            left_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u16
                - (board_red.get_raw().count_ones() + board_blue.get_raw().count_ones()) as u16,
            winner: None,
        };

        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if let Some(color) = position.get_tile(r, c) {
                    position.update_reach(r, c, color);
                }
            }
        }

        position
    }

    pub fn pieces_red(&self) -> HexBitboard<BOARD_SIZE> {
        self.board_red
    }

    pub fn pieces_blue(&self) -> HexBitboard<BOARD_SIZE> {
        self.board_blue
    }

    pub fn is_valid_move(&self, m: HexMove<BOARD_SIZE>) -> bool {
        if self.status().is_finished() {
            return false;
        }
        let idx = m.to_idx();
        !self.board_red.get(idx) && !self.board_blue.get(idx)
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        let idx = r * BOARD_SIZE + c;
        if self.board_red.get(idx) {
            Some(GameColor::Player1)
        } else if self.board_blue.get(idx) {
            Some(GameColor::Player2)
        } else {
            None
        }
    }

    fn neighbors(r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
        let connection_dirs = [
            /*      right */ (0, 1),
            /* up   right */ (-1, 1),
            /* up   left  */ (-1, 0),
            /*      left  */ (0, -1),
            /* down left  */ (1, -1),
            /* down right */ (1, 0),
        ];
        connection_dirs
            .into_iter()
            .map(move |(dr, dc)| (r as i8 + dr, c as i8 + dc))
            .filter(|(nr, nc)| (0..BOARD_SIZE as i8).contains(nr) && (0..BOARD_SIZE as i8).contains(nc))
            .map(|(nr, nc)| (nr as usize, nc as usize))
    }

    fn update_reach(&mut self, r: usize, c: usize, player: GameColor) {
        let reach_map = match player {
            GameColor::Player1 => &mut self.top_red_reach,
            GameColor::Player2 => &mut self.left_blue_reach,
        };
        let is_reach_begin = match player {
            GameColor::Player1 => |r: usize, _: usize| r == 0,
            GameColor::Player2 => |_: usize, c: usize| c == 0,
        };
        if !(is_reach_begin(r, c) || Self::neighbors(r, c).any(|(nr, nc)| reach_map.get(nr * BOARD_SIZE + nc))) {
            return;
        }

        let board = match player {
            GameColor::Player1 => &self.board_red,
            GameColor::Player2 => &self.board_blue,
        };
        let is_reach_end = match player {
            GameColor::Player1 => |r: usize, _: usize| r == BOARD_SIZE - 1,
            GameColor::Player2 => |_: usize, c: usize| c == BOARD_SIZE - 1,
        };

        let mut stack = HexBitboard::<BOARD_SIZE>::new();
        let idx = r * BOARD_SIZE + c;
        reach_map.set(idx, true);
        stack.set(idx, true);

        while !stack.is_empty() {
            let idx = stack.get_raw().trailing_zeros() as usize;
            stack.set(idx, false);
            let (r, c) = (idx / BOARD_SIZE, idx % BOARD_SIZE);

            if is_reach_end(r, c) {
                self.winner = Some(player);
            }

            for (nr, nc) in Self::neighbors(r, c) {
                let n_idx = nr * BOARD_SIZE + nc;
                if board.get(n_idx) && !reach_map.get(n_idx) {
                    reach_map.set(n_idx, true);
                    stack.set(n_idx, true);
                }
            }
        }
    }

    pub fn make_move(&mut self, m: HexMove<BOARD_SIZE>) {
        assert!(self.is_valid_move(m));

        match self.turn {
            GameColor::Player1 => &mut self.board_red,
            GameColor::Player2 => &mut self.board_blue,
        }
        .set(m.to_idx(), true);

        self.update_reach(m.row(), m.column(), self.turn);

        self.number_of_empty_tiles -= 1;
        self.turn = self.turn.opposite();
    }
}

impl<const BOARD_SIZE: usize> Display for HexPosition<BOARD_SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..BOARD_SIZE {
            for _ in 0..r {
                write!(f, " ")?;
            }
            for c in 0..BOARD_SIZE {
                let ch = match self.get_tile(r, c) {
                    Some(GameColor::Player1) => 'R',
                    Some(GameColor::Player2) => 'B',
                    None => '.',
                };
                write!(f, "{ch} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<const BOARD_SIZE: usize> Position for HexPosition<BOARD_SIZE> {
    type Game = HexGame<BOARD_SIZE>;
    fn new() -> Self {
        HexPosition::new_with_starting_color(GameColor::Player1)
    }
    fn turn(&self) -> GameColor {
        self.turn
    }

    fn legal_moves(&self) -> impl Iterator<Item = HexMove<BOARD_SIZE>> {
        (0..(BOARD_SIZE * BOARD_SIZE))
            .filter(|&idx| !self.board_red.get(idx) && !self.board_blue.get(idx))
            .map(HexMove::from_idx)
    }

    fn moved_position(&self, m: HexMove<BOARD_SIZE>) -> Self {
        let mut res = *self;
        res.make_move(m);
        res
    }

    fn status(&self) -> GameStatus {
        if self.winner.is_some() {
            GameStatus::Finished(self.winner)
        } else if self.number_of_empty_tiles == 0 {
            GameStatus::Finished(None)
        } else {
            GameStatus::Ongoing
        }
    }

    fn flipped(&self) -> Self {
        Self {
            board_red: self.board_blue.flip(),
            board_blue: self.board_red.flip(),
            turn: self.turn.opposite(),
            top_red_reach: self.left_blue_reach.flip(),
            left_blue_reach: self.top_red_reach.flip(),
            number_of_empty_tiles: self.number_of_empty_tiles,
            winner: self.winner.map(|w| w.opposite()),
        }
    }
}

pub struct HexGame<const BOARD_SIZE: usize> {
    pos_history: Vec<HexPosition<BOARD_SIZE>>,
}

pub const HEX_STANDARD_BOARD_SIZE: usize = 11;
pub type HexGameStandard = HexGame<HEX_STANDARD_BOARD_SIZE>;

impl<const BOARD_SIZE: usize> Game for HexGame<BOARD_SIZE> {
    type Position = HexPosition<BOARD_SIZE>;
    type Move = HexMove<BOARD_SIZE>;
    type Bitboard = HexBitboard<BOARD_SIZE>;
    const BOARD_SIZE: usize = BOARD_SIZE;
    const MOVES_NUM: usize = BOARD_SIZE * BOARD_SIZE;
    const REPETITION_LIMIT: Option<usize> = None;

    fn new() -> Self {
        Self::from_position(HexPosition::new())
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
    use itertools::Itertools;
    use rand::prelude::*;
    use std::collections::HashSet;

    use crate::game::player::{GamePlayer, PlayerRand};
    use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};
    use crate::hex::{HexBitboard, HexGameStandard, HexMove, HexPosition};

    type HexStandardPosition = <HexGameStandard as Game>::Position;

    #[test]
    fn short_diagonal_wins() {
        let pos: HexStandardPosition = hex_position_from_str(
            ". . . . . . . . . . r\
              . . . . . . . . . r .\
               . . . . . . . . r . .\
                . . . . . . . r . . .\
                 . . . . . . r . . . .\
                  . . . . . r . . . . .\
                   . . . . r . . . . . .\
                    . . . r . . . . . . .\
                     . . r . . . . . . . .\
                      . r . . . . . . . . .\
                       r . . . . . . . . . .\
                b",
        );
        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player1)));

        let pos: HexStandardPosition = hex_position_from_str(
            ". . . . . . . . . . b\
              . . . . . . . . . b .\
               . . . . . . . . b . .\
                . . . . . . . b . . .\
                 . . . . . . b . . . .\
                  . . . . . b . . . . .\
                   . . . . b . . . . . .\
                    . . . b . . . . . . .\
                     . . b . . . . . . . .\
                      . b . . . . . . . . .\
                       b . . . . . . . . . .\
            r",
        );
        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player2)));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
            ". . . . . . . . . . r\
              . . . . . . . . . r .\
               . . . . . . . . r . .\
                . . . . . . . . . . .\
                 . . . . . . r . . . .\
                  . . . . . r . . . . .\
                   . . . . r . . . . . .\
                    . . . r . . . . . . .\
                     . . r . . . . . . . .\
                      . r . . . . . . . . .\
                       r . . . . . . . . . .\
            b",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);

        let pos: HexStandardPosition = hex_position_from_str(
            ". . . . . . . . . . b\
              . . . . . . . . . b .\
               . . . . . . . . b . .\
                . . . . . . . b . . .\
                 . . . . . . b . . . .\
                  . . . . . b . . . . .\
                   . . . . b . . . . . .\
                    . . . . . . . . . . .\
                     . . b . . . . . . . .\
                      . b . . . . . . . . .\
                       b . . . . . . . . . .\
            r",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
            "r . . . . . . . . . .\
              . r . . . . . . . . .\
               . . r . . . . . . . .\
                . . . r . . . . . . .\
                 . . . . r . . . . . .\
                  . . . . . r . . . . .\
                   . . . . . . r . . . .\
                    . . . . . . . r . . .\
                     . . . . . . . . r . .\
                      . . . . . . . . . r .\
                       . . . . . . . . . . r\
            b",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);

        let pos: HexStandardPosition = hex_position_from_str(
            "b . . . . . . . . . .\
              . b . . . . . . . . .\
               . . b . . . . . . . .\
                . . . b . . . . . . .\
                 . . . . b . . . . . .\
                  . . . . . b . . . . .\
                   . . . . . . b . . . .\
                    . . . . . . . b . . .\
                     . . . . . . . . b . .\
                      . . . . . . . . . b .\
                       . . . . . . . . . . b\
            r",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);
    }

    #[test]
    fn board4() {
        let mut rand = StdRng::seed_from_u64(0xb843ecbdea516a01);

        let red_wins = [
            ". . . r
              . . r .
               . r . .
                r . . .",
            ". . r .
              . r . .
               . r . .
                . r . .",
            "r . . .
              r . . .
               r . . .
                r . . .",
            ". . . r
              . . . r
               . . . r
                . . . r",
            ". . r .
              . r . .
               . r . .
                r . . .",
        ];
        for board_str in red_wins {
            let pos = hex_position_from_str::<4>(&format!("{board_str}b"));
            assert_eq!(
                pos.status(),
                GameStatus::Finished(Some(GameColor::Player1)),
                "board:\n{pos}"
            );

            for _ in 0..100 {
                let red = pos.pieces_red();
                let mut blue = pos.pieces_blue();
                for idx in 0..16 {
                    if !red.get(idx) && !blue.get(idx) && rand.random::<bool>() {
                        blue.set(idx, true);
                    }
                }
                let pos = HexPosition::new_from_board(red, blue, GameColor::Player2);
                assert_eq!(
                    pos.status(),
                    GameStatus::Finished(Some(GameColor::Player1)),
                    "board:\n{pos}"
                );

                let mut red_indices = pos.pieces_red();
                while !red_indices.is_empty() {
                    let idx = red_indices.get_raw().trailing_zeros() as usize;
                    red_indices.set(idx, false);

                    let mut red = pos.pieces_red();
                    let blue = pos.pieces_blue();
                    red.set(idx, false);
                    let pos = HexPosition::new_from_board(red, blue, GameColor::Player1);
                    assert_eq!(pos.status(), GameStatus::Ongoing, "board:\n{pos}");

                    let pos = pos.moved_position(HexMove::from_idx(idx));
                    assert_eq!(
                        pos.status(),
                        GameStatus::Finished(Some(GameColor::Player1)),
                        "board:\n{pos}"
                    );
                }
            }
        }

        let blue_wins = [
            ". . . b
              . . b .
               . b . .
                b . . .",
            ". . . .
              . . . b
               . b b .
                b . . .",
            ". . . .
              b b b b
               . . . .
                . . . .",
            "b b b .
              . . b .
               . . b b
                . . . .",
            ". . . b
              . . b .
               . . b .
                b b . .",
        ];
        for board_str in blue_wins {
            let pos = hex_position_from_str::<4>(&format!("{board_str}r"));
            assert_eq!(
                pos.status(),
                GameStatus::Finished(Some(GameColor::Player2)),
                "board:\n{pos}"
            );

            for _ in 0..100 {
                let blue = pos.pieces_blue();
                let mut red = pos.pieces_red();
                for idx in 0..16 {
                    if !red.get(idx) && !blue.get(idx) && rand.random::<bool>() {
                        red.set(idx, true);
                    }
                }
                let pos = HexPosition::new_from_board(red, blue, GameColor::Player1);
                assert_eq!(
                    pos.status(),
                    GameStatus::Finished(Some(GameColor::Player2)),
                    "board:\n{pos}"
                );

                let mut blue_indices = pos.pieces_blue();
                while !blue_indices.is_empty() {
                    let idx = blue_indices.get_raw().trailing_zeros() as usize;
                    blue_indices.set(idx, false);

                    let red = pos.pieces_red();
                    let mut blue = pos.pieces_blue();
                    blue.set(idx, false);
                    let pos = HexPosition::new_from_board(red, blue, GameColor::Player2);
                    assert_eq!(pos.status(), GameStatus::Ongoing, "board:\n{pos}");

                    let pos = pos.moved_position(HexMove::from_idx(idx));
                    assert_eq!(
                        pos.status(),
                        GameStatus::Finished(Some(GameColor::Player2)),
                        "board:\n{pos}"
                    );
                }
            }
        }
    }

    #[test]
    fn flip() {
        let pos = hex_position_from_str::<4>(
            ". . r b
              . b r .
               b . r .
                . r . .
            b",
        );

        assert_eq!(pos.turn(), GameColor::Player2);
        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player1)));

        assert_eq!(pos.flipped().turn(), GameColor::Player1);
        assert_eq!(pos.flipped().status(), GameStatus::Finished(Some(GameColor::Player2)));

        assert_eq!(pos.flipped().flipped(), pos);

        let mut red_indices = pos.pieces_red();
        while !red_indices.is_empty() {
            let idx = red_indices.get_raw().trailing_zeros() as usize;
            red_indices.set(idx, false);

            let mut red = pos.pieces_red();
            let blue = pos.pieces_blue();
            red.set(idx, false);
            let pos = HexPosition::new_from_board(red, blue, GameColor::Player1);
            assert_eq!(pos.status(), GameStatus::Ongoing);
            let move_ = HexMove::from_idx(idx);

            let f_pos = pos.flipped();
            assert_eq!(f_pos.status(), GameStatus::Ongoing);
            let f_pos = f_pos.moved_position(move_.flipped());
            assert_eq!(f_pos.status(), GameStatus::Finished(Some(GameColor::Player2)));

            assert_eq!(pos.moved_position(move_).flipped(), f_pos);
            assert_eq!(pos.moved_position(move_), f_pos.flipped());
        }
    }

    #[test]
    fn flip_rand() {
        let mut rand = StdRng::seed_from_u64(0x8e931b83f015b328);

        let games_num = 100;
        for _ in 0..games_num {
            let mut player = PlayerRand::from_seed(rand.next_u64() ^ 0x669d82f7a78d1f5);
            let mut game = HexGameStandard::new();

            while game.status().is_ongoing() {
                let pos = *game.position();
                let pos_t = pos.flipped();

                /* Assert flip of flip is original */
                assert_eq!(pos, pos_t.flipped());

                /* Assert flip of moves of flip are original moves */
                let moves = pos.legal_moves().collect::<HashSet<_>>();
                let moves_tt = pos_t.legal_moves().map(|m| m.flipped()).collect::<HashSet<_>>();
                assert_eq!(moves, moves_tt);

                /* Assert game result is the same */
                match (pos.status(), pos_t.status()) {
                    (GameStatus::Finished(c1), GameStatus::Finished(c2)) => assert_eq!(c1, c2.map(|c| c.opposite())),
                    (GameStatus::Ongoing, GameStatus::Ongoing) => {}
                    _ => panic!("One game ended but not the other"),
                }

                let next_move = <_ as GamePlayer<HexGameStandard>>::next_move(&mut player, game.pos_history()).unwrap();
                game.play_single_turn(next_move);
            }
        }
    }

    pub fn hex_position_from_str<const BOARD_SIZE: usize>(s: &str) -> HexPosition<BOARD_SIZE> {
        let s = s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        assert_eq!(s.len(), BOARD_SIZE * BOARD_SIZE + 1, "unexpected string length");
        let lines = s
            .chars()
            .chunks(BOARD_SIZE)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let board_lines = &lines[..BOARD_SIZE];
        let last_line = &lines[BOARD_SIZE];

        let mut board_red = HexBitboard::new();
        let mut board_blue = HexBitboard::new();
        for (row, line) in board_lines.iter().enumerate() {
            for (col, c) in line.iter().enumerate() {
                match c {
                    'e' | '.' => {}
                    'r' => board_red.set(row * BOARD_SIZE + col, true),
                    'b' => board_blue.set(row * BOARD_SIZE + col, true),
                    _ => panic!("unknown board char: {c:?}"),
                }
            }
        }

        let turn = match last_line[0] {
            'r' => GameColor::Player1,
            'b' => GameColor::Player2,
            unknown_turn_char => panic!("unknown turn char: {unknown_turn_char:?}"),
        };

        HexPosition::new_from_board(board_red, board_blue, turn)
    }
}
