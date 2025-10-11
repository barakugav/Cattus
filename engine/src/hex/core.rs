use std::fmt::{self, Display};

use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexMove<const BOARD_SIZE: usize> {
    idx: u8,
}

impl<const BOARD_SIZE: usize> HexMove<BOARD_SIZE> {
    pub fn new(r: usize, c: usize) -> Self {
        HexMove::from_idx(r * BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        Self { idx: idx as u8 }
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
    /// The board is a rhombus, slanted right. So, board[0][BOARD_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    pub board_red: HexBitboard<BOARD_SIZE>,
    pub board_blue: HexBitboard<BOARD_SIZE>,
    pub turn: GameColor,

    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with red pieces */
    left_red_reach: HexBitboard<BOARD_SIZE>,
    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with blue pieces */
    top_blue_reach: HexBitboard<BOARD_SIZE>,
    number_of_empty_tiles: u8,
    winner: Option<GameColor>,
}

impl<const BOARD_SIZE: usize> HexPosition<BOARD_SIZE> {
    pub fn new_with_starting_color(starting_color: GameColor) -> Self {
        Self {
            board_red: HexBitboard::new(),
            board_blue: HexBitboard::new(),
            turn: starting_color,
            left_red_reach: HexBitboard::new(),
            top_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        }
    }

    pub fn new_from_board(
        board_red: HexBitboard<BOARD_SIZE>,
        board_blue: HexBitboard<BOARD_SIZE>,
        turn: GameColor,
    ) -> Self {
        let mut s = Self {
            board_red,
            board_blue,
            turn,
            left_red_reach: HexBitboard::new(),
            top_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        };

        let is_reach_begin = |r: usize, c: usize, player: GameColor| match player {
            GameColor::Player1 => c == 0,
            GameColor::Player2 => r == 0,
        };
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if let Some(color) = s.get_tile(r, c) {
                    s.number_of_empty_tiles -= 1;
                    if is_reach_begin(r, c, color) {
                        s.update_reach(r, c, color);
                    }
                }
            }
        }
        s
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
        idx < BOARD_SIZE * BOARD_SIZE && !self.board_red.get(idx) && !self.board_blue.get(idx)
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        let idx = r * BOARD_SIZE + c;
        if self.board_red.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_blue.get(idx) {
            return Some(GameColor::Player2);
        }
        None
    }

    fn foreach_neighbor<OP: FnMut(usize, usize)>(r: usize, c: usize, mut op: OP) {
        let connection_dirs: [(i8, i8); 6] = [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1)];
        for (dr, dc) in connection_dirs {
            let nr = r as i8 + dr;
            let nc = c as i8 + dc;
            if nr < 0 || nr as usize >= BOARD_SIZE || nc < 0 || nc as usize >= BOARD_SIZE {
                continue;
            }
            op(nr as usize, nc as usize);
        }
    }

    fn update_reach(&mut self, r: usize, c: usize, player: GameColor) {
        let board = match player {
            GameColor::Player1 => &self.board_red,
            GameColor::Player2 => &self.board_blue,
        };
        let reach_map = match player {
            GameColor::Player1 => &mut self.left_red_reach,
            GameColor::Player2 => &mut self.top_blue_reach,
        };
        let is_reach_begin = match player {
            GameColor::Player1 => |_: usize, c: usize| c == 0,
            GameColor::Player2 => |r: usize, _: usize| r == 0,
        };
        let is_reach_end = match player {
            GameColor::Player1 => |_: usize, c: usize| c == BOARD_SIZE - 1,
            GameColor::Player2 => |r: usize, _: usize| r == BOARD_SIZE - 1,
        };

        let mut bfs_layer = HexBitboard::<BOARD_SIZE>::new();

        let mut update_reach = is_reach_begin(r, c);
        Self::foreach_neighbor(r, c, |nr: usize, nc: usize| {
            let n_idx = nr * BOARD_SIZE + nc;
            update_reach = update_reach || reach_map.get(n_idx);
        });
        if update_reach {
            let idx = r * BOARD_SIZE + c;
            reach_map.set(idx, true);
            bfs_layer.set(idx, true);
        }

        while !bfs_layer.is_empty() {
            let idx = bfs_layer.get_raw().trailing_zeros() as usize;
            bfs_layer.set(idx, false);
            let r = idx / BOARD_SIZE;
            let c = idx % BOARD_SIZE;

            if is_reach_end(r, c) {
                self.winner = Some(player);
            } else {
                Self::foreach_neighbor(r, c, |nr: usize, nc: usize| {
                    let n_idx = nr * BOARD_SIZE + nc;
                    if !reach_map.get(n_idx) && board.get(n_idx) {
                        reach_map.set(n_idx, true);
                        bfs_layer.set(n_idx, true);
                    }
                });
            }
        }
    }

    pub fn make_move_new(&self, m: HexMove<BOARD_SIZE>) -> Self {
        let mut res = *self;
        res.make_move(m);
        res
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

impl<const BOARD_SIZE: usize> Position for HexPosition<BOARD_SIZE> {
    type Game = HexGame<BOARD_SIZE>;
    fn new() -> Self {
        HexPosition::new_with_starting_color(GameColor::Player1)
    }
    fn turn(&self) -> GameColor {
        self.turn
    }

    fn legal_moves(&self) -> impl Iterator<Item = HexMove<BOARD_SIZE>> {
        (0..(BOARD_SIZE * BOARD_SIZE)).filter_map(|idx| {
            if !self.board_red.get(idx) && !self.board_blue.get(idx) {
                Some(HexMove::from_idx(idx))
            } else {
                None
            }
        })
    }

    fn moved_position(&self, m: HexMove<BOARD_SIZE>) -> Self {
        assert!(self.is_valid_move(m));
        let mut res = *self;
        res.make_move(m);
        res
    }

    fn status(&self) -> GameStatus {
        if self.winner.is_some() {
            return GameStatus::Finished(self.winner);
        }
        if self.number_of_empty_tiles == 0 {
            return GameStatus::Finished(None);
        }
        GameStatus::Ongoing
    }

    fn flipped(&self) -> Self {
        Self {
            board_red: self.board_blue.flip(),
            board_blue: self.board_red.flip(),
            turn: self.turn.opposite(),
            left_red_reach: self.top_blue_reach.flip(),
            top_blue_reach: self.left_red_reach.flip(),
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
        self.pos_history.push(self.position().make_move_new(next_move));
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    use crate::game::player::{GamePlayer, PlayerRand};
    use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};
    use crate::hex::{HexBitboard, HexGameStandard, HexPosition};

    type HexStandardPosition = <HexGameStandard as Game>::Position;

    #[test]
    fn short_diagonal_wins() {
        let pos: HexStandardPosition = hex_position_from_str(
            "reeeeeeeeee\
            ereeeeeeeee\
            eereeeeeeee\
            eeereeeeeee\
            eeeereeeeee\
            eeeeereeeee\
            eeeeeereeee\
            eeeeeeereee\
            eeeeeeeeree\
            eeeeeeeeere\
            eeeeeeeeeer\
            b",
        );
        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player1)));

        let pos: HexStandardPosition = hex_position_from_str(
            "beeeeeeeeee\
            ebeeeeeeeee\
            eebeeeeeeee\
            eeebeeeeeee\
            eeeebeeeeee\
            eeeeebeeeee\
            eeeeeebeeee\
            eeeeeeebeee\
            eeeeeeeebee\
            eeeeeeeeebe\
            eeeeeeeeeeb\
            r",
        );

        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player2)));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
            "eeeeeeeeeee\
            ereeeeeeeee\
            eereeeeeeee\
            eeereeeeeee\
            eeeereeeeee\
            eeeeereeeee\
            eeeeeereeee\
            eeeeeeereee\
            eeeeeeeeree\
            eeeeeeeeere\
            eeeeeeeeeer\
            b",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);

        let pos: HexStandardPosition = hex_position_from_str(
            "beeeeeeeeee\
            ebeeeeeeeee\
            eebeeeeeeee\
            eeebeeeeeee\
            eeeebeeeeee\
            eeeeebeeeee\
            eeeeeebeeee\
            eeeeeeebeee\
            eeeeeeeebee\
            eeeeeeeeebe\
            eeeeeeeeeee\
            r",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
            "eeeeeeeeeer\
            eeeeeeeeere\
            eeeeeeeeree\
            eeeeeeereee\
            eeeeeereeee\
            eeeeereeeee\
            eeeereeeeee\
            eeereeeeeee\
            eereeeeeeee\
            ereeeeeeeee\
            reeeeeeeeee\
            b",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);

        let pos: HexStandardPosition = hex_position_from_str(
            "eeeeeeeeeeb\
            eeeeeeeeebe\
            eeeeeeeebee\
            eeeeeeebeee\
            eeeeeebeeee\
            eeeeebeeeee\
            eeeebeeeeee\
            eeebeeeeeee\
            eebeeeeeeee\
            ebeeeeeeeee\
            beeeeeeeeee\
            r",
        );
        assert_eq!(pos.status(), GameStatus::Ongoing);
    }

    #[test]
    fn flip() {
        let pos: HexStandardPosition = hex_position_from_str(
            "eebeeeeeeer\
            eeeeeeeeeee\
            eeeebeeeree\
            eeeeeeereee\
            eeeeeereeee\
            eeeeereeeee\
            eeeerebeeee\
            eeereeeeeee\
            eereeereeee\
            ereeeeeeeee\
            reeeeebeeee\
            b",
        );
        assert_eq!(pos.turn(), GameColor::Player2);
        assert_eq!(pos.flipped().turn(), GameColor::Player1);
        assert_eq!(pos.flipped().flipped(), pos);
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
                type Move = <HexGameStandard as Game>::Move;
                let moves: HashSet<Move> = HashSet::from_iter(pos.legal_moves());
                let moves_tt: HashSet<Move> = HashSet::from_iter(pos_t.legal_moves().into_iter().map(|m| m.flipped()));
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
        assert_eq!(
            s.chars().count(),
            BOARD_SIZE * BOARD_SIZE + 1,
            "unexpected string length"
        );

        let mut board_red = HexBitboard::new();
        let mut board_blue = HexBitboard::new();
        let mut turn = None;
        for (idx, c) in s.chars().enumerate() {
            match idx.cmp(&(BOARD_SIZE * BOARD_SIZE)) {
                Ordering::Less => match c {
                    'e' => {}
                    'r' => board_red.set(idx, true),
                    'b' => board_blue.set(idx, true),
                    _ => panic!("unknown board char: {:?}", c),
                },
                Ordering::Equal => {
                    turn = Some(match c {
                        'r' => GameColor::Player1,
                        'b' => GameColor::Player2,
                        _ => panic!("unknown turn char: {:?}", c),
                    })
                }
                Ordering::Greater => panic!("Too many chars in position string"),
            }
        }

        HexPosition::new_from_board(board_red, board_blue, turn.unwrap())
    }
}
