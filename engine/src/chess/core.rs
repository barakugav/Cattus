use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use crate::game::{Bitboard, Game, GameColor, GameStatus, Move, Position};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChessMove {
    m: chess::ChessMove,
}

impl ChessMove {
    pub fn new(m: chess::ChessMove) -> Self {
        Self { m }
    }

    pub fn from_san(pos: &ChessPosition, move_str: &str) -> Result<Self, chess::Error> {
        chess::ChessMove::from_san(&pos.board, move_str).map(Self::new)
    }

    pub fn from_lan(move_str: &str) -> Result<Self, String> {
        if !(move_str.len() == 4 || move_str.len() == 5) {
            return Err(format!("Invalid LAN length: '{move_str}'"));
        }
        #[allow(clippy::iter_nth_zero)]
        let sf = chess::File::from_index(move_str.chars().nth(0).unwrap() as usize - 'a' as usize);
        let sr = chess::Rank::from_index(move_str.chars().nth(1).unwrap() as usize - '1' as usize);
        let df = chess::File::from_index(move_str.chars().nth(2).unwrap() as usize - 'a' as usize);
        let dr = chess::Rank::from_index(move_str.chars().nth(3).unwrap() as usize - '1' as usize);
        let source = chess::Square::make_square(sr, sf);
        let dest = chess::Square::make_square(dr, df);
        let promotion = if move_str.len() == 5 {
            Some(match move_str.chars().nth(4).unwrap() {
                'q' => chess::Piece::Queen,
                'n' => chess::Piece::Knight,
                'r' => chess::Piece::Rook,
                'b' => chess::Piece::Bishop,
                c => {
                    return Err(format!("Unknown promotion char: '{c}' in lan str '{move_str}'"));
                }
            })
        } else {
            None
        };
        Ok(ChessMove::new(chess::ChessMove::new(source, dest, promotion)))
    }

    pub fn get_raw(&self) -> &chess::ChessMove {
        &self.m
    }

    pub fn to_idx(&self) -> usize {
        if let Some(promotion) = self.m.get_promotion() {
            let promotions_base = 64 * 64;
            let sf = self.m.get_source().get_file().to_index();
            let df = self.m.get_dest().get_file().to_index();
            let move_chunk = (sf * 2 + df) * 4;
            let piece_offset = match promotion {
                chess::Piece::Queen => 0,
                chess::Piece::Rook => 1,
                chess::Piece::Bishop => 2,
                chess::Piece::Knight => 3,
                other => panic!("unexpected promotion piece: {other:?}"),
            };
            promotions_base + move_chunk + piece_offset
        } else {
            self.m.get_source().to_index() * 64 + self.m.get_dest().to_index()
        }
    }
}
impl AsRef<chess::ChessMove> for ChessMove {
    fn as_ref(&self) -> &chess::ChessMove {
        &self.m
    }
}
impl Move for ChessMove {
    type Game = ChessGame;

    fn flipped(&self) -> Self {
        let flip_rank = |r: chess::Rank| chess::Rank::from_index(ChessGame::BOARD_SIZE - 1 - r.to_index());
        let flip_file = |f: chess::File| f;
        let flip_square =
            |s: chess::Square| chess::Square::make_square(flip_rank(s.get_rank()), flip_file(s.get_file()));
        let s = flip_square(self.m.get_source());
        let d = flip_square(self.m.get_dest());
        let promotion = self.m.get_promotion();
        ChessMove::new(chess::ChessMove::new(s, d, promotion))
    }

    fn to_nn_idx(&self) -> usize {
        MOVE_TO_NN_INDEX[self.to_idx()] as usize
    }
}

impl Display for ChessMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.m.fmt(f)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChessBitboard {
    b: chess::BitBoard,
}

impl ChessBitboard {
    pub fn from_raw(bitmap: u64) -> Self {
        Self {
            b: chess::BitBoard::new(bitmap),
        }
    }

    pub fn get_raw(&self) -> u64 {
        self.b.0
    }
}

impl Bitboard for ChessBitboard {
    type Game = ChessGame;
    fn new() -> Self {
        Self {
            b: chess::BitBoard::new(0),
        }
    }

    fn full(val: bool) -> Self {
        Self {
            b: chess::BitBoard::new(if val { ((1u128 << 64) - 1) as u64 } else { 0 }),
        }
    }

    fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < ChessGame::BOARD_SIZE * ChessGame::BOARD_SIZE);
        (self.b.0 & (1u64 << idx)) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        debug_assert!(idx < ChessGame::BOARD_SIZE * ChessGame::BOARD_SIZE);
        if val {
            self.b.0 |= 1u64 << idx;
        } else {
            self.b.0 &= !(1u64 << idx);
        }
    }
}
impl From<chess::BitBoard> for ChessBitboard {
    fn from(b: chess::BitBoard) -> Self {
        Self::from_raw(b.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ChessPosition {
    pub board: chess::Board,
    pub fifty_rule_count: u8,
}

impl ChessPosition {
    fn new_from_board(board: chess::Board) -> Self {
        Self {
            board,
            fifty_rule_count: 0,
        }
    }

    pub fn from_fen(s: &str) -> Self {
        Self::new_from_board(chess::Board::from_str(s).unwrap())
    }

    pub fn is_valid_move(&self, m: ChessMove) -> bool {
        self.status().is_ongoing() && self.board.legal(m.m)
    }

    pub fn get_raw_board(&self) -> &chess::Board {
        &self.board
    }

    pub fn fen(&self) -> String {
        let b = &self.board;
        let mut s = String::new();

        for reverse_rnk in 0..ChessGame::BOARD_SIZE {
            let mut blanks = 0;
            let rank = ChessGame::BOARD_SIZE - 1 - reverse_rnk;

            if reverse_rnk != 0 {
                s.push('/');
            }

            for file in 0..ChessGame::BOARD_SIZE {
                let sq = chess::Square::make_square(chess::Rank::from_index(rank), chess::File::from_index(file));
                match b.piece_on(sq) {
                    Some(piece) => {
                        if blanks != 0 {
                            s.push(char::from_digit(blanks, 10).unwrap());
                            blanks = 0;
                        }
                        let mut c = match piece {
                            chess::Piece::Pawn => 'P',
                            chess::Piece::Knight => 'N',
                            chess::Piece::Bishop => 'B',
                            chess::Piece::Rook => 'R',
                            chess::Piece::Queen => 'Q',
                            chess::Piece::King => 'K',
                        };
                        if b.color_on(sq).unwrap() == chess::Color::Black {
                            c = c.to_lowercase().next().unwrap();
                        }
                        s.push(c);
                    }
                    None => blanks += 1,
                }
            }
            if blanks != 0 {
                s.push(char::from_digit(blanks, 10).unwrap());
            }
        }

        s.push(' ');
        // current turn
        s.push(match b.side_to_move() {
            chess::Color::White => 'w',
            chess::Color::Black => 'b',
        });
        s.push(' ');

        // Castling State
        let cr_w = b.castle_rights(chess::Color::White);
        let cr_b = b.castle_rights(chess::Color::Black);
        let cr_str = if cr_w == chess::CastleRights::NoRights && cr_b == chess::CastleRights::NoRights {
            "-".to_string()
        } else {
            let mut s = String::new();
            if cr_w.has_kingside() {
                s.push('K');
            }
            if cr_w.has_queenside() {
                s.push('Q');
            }

            if cr_b.has_kingside() {
                s.push('k');
            }

            if cr_b.has_queenside() {
                s.push('q');
            }
            assert!(!s.is_empty());
            s
        };
        s.push_str(&cr_str);
        s.push(' ');

        // EP Square
        match b.en_passant() {
            Some(sq) => {
                let rank_idx = (sq.get_rank().to_index() as i64
                    /* The 'chess' library return the square of the pawn, while FEN */
                    /* include the square over which a pawn has just passed */
                    + match b.side_to_move() {
                        chess::Color::White => 1,
                        chess::Color::Black => -1,
                    }) as usize;
                s.push(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][sq.get_file().to_index()]);
                s.push(['1', '2', '3', '4', '5', '6', '7', '8'][rank_idx]);
            }
            None => s.push('-'),
        }
        // s.push(' ');
        // s.push_str(&format!("{}", self.rule_50()));
        // s.push(' ');
        // s.push_str(&format!("{}", (self.half_moves / 2) + 1));

        s
    }
}

fn chess_color_to_game_color(c: chess::Color) -> GameColor {
    match c {
        chess::Color::White => GameColor::Player1,
        chess::Color::Black => GameColor::Player2,
    }
}

impl PartialEq for ChessPosition {
    fn eq(&self, other: &Self) -> bool {
        let b1 = &self.board;
        let b2 = &other.board;
        b1.color_combined(chess::Color::White) == b2.color_combined(chess::Color::White)
            && b1.color_combined(chess::Color::Black) == b2.color_combined(chess::Color::Black)
            && b1.pieces(chess::Piece::Pawn) == b2.pieces(chess::Piece::Pawn)
            && b1.pieces(chess::Piece::Knight) == b2.pieces(chess::Piece::Knight)
            && b1.pieces(chess::Piece::Bishop) == b2.pieces(chess::Piece::Bishop)
            && b1.pieces(chess::Piece::Rook) == b2.pieces(chess::Piece::Rook)
            && b1.pieces(chess::Piece::Queen) == b2.pieces(chess::Piece::Queen)
            && b1.pieces(chess::Piece::King) == b2.pieces(chess::Piece::King)
            && b1.castle_rights(chess::Color::White) == b2.castle_rights(chess::Color::White)
            && b1.castle_rights(chess::Color::Black) == b2.castle_rights(chess::Color::Black)
            && b1.en_passant() == b2.en_passant()
            && b1.side_to_move() == b2.side_to_move()
    }
}
impl Eq for ChessPosition {}
impl Hash for ChessPosition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.get_hash().hash(state);
    }
}

impl Display for ChessPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in (0..ChessGame::BOARD_SIZE).rev() {
            for file in 0..ChessGame::BOARD_SIZE {
                let sq = chess::Square::make_square(chess::Rank::from_index(r), chess::File::from_index(file));
                match self.board.piece_on(sq) {
                    Some(piece) => {
                        let mut c = match piece {
                            chess::Piece::Pawn => 'P',
                            chess::Piece::Knight => 'N',
                            chess::Piece::Bishop => 'B',
                            chess::Piece::Rook => 'R',
                            chess::Piece::Queen => 'Q',
                            chess::Piece::King => 'K',
                        };
                        if self.board.color_on(sq).unwrap() == chess::Color::Black {
                            c = c.to_lowercase().next().unwrap();
                        }
                        write!(f, "{c} ")?;
                    }
                    None => {
                        write!(f, ". ")?;
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Position for ChessPosition {
    type Game = ChessGame;
    fn new() -> Self {
        Self::new_from_board(chess::Board::default())
    }

    fn turn(&self) -> GameColor {
        chess_color_to_game_color(self.board.side_to_move())
    }

    fn legal_moves(&self) -> impl Iterator<Item = ChessMove> {
        chess::MoveGen::new_legal(&self.board).map(ChessMove::new)
    }

    fn moved_position(&self, m: ChessMove) -> Self {
        assert!(self.is_valid_move(m));

        let mut next_board = ChessPosition::new_from_board(self.board.make_move_new(m.m));

        let piece = self.board.piece_on(m.m.get_source());
        let is_pawn = piece == Some(chess::Piece::Pawn);
        let is_atk = self.board.piece_on(m.m.get_dest()).is_some();

        next_board.fifty_rule_count = if is_pawn || is_atk {
            0
        } else if self.turn() == GameColor::Player1 {
            /* The move count is incremented only once per white+black move */
            self.fifty_rule_count + 1
        } else {
            self.fifty_rule_count
        };

        next_board
    }

    fn status(&self) -> GameStatus {
        if self.board.status() == chess::BoardStatus::Ongoing && self.fifty_rule_count < 50 {
            return GameStatus::Ongoing;
        }
        GameStatus::Finished(match self.board.status() {
            chess::BoardStatus::Ongoing => {
                assert!(self.fifty_rule_count >= 50);
                None
            }
            chess::BoardStatus::Stalemate => None,
            chess::BoardStatus::Checkmate => {
                // TODO not sure this is correct, need to check.
                // looks valid according to https://docs.rs/chess/latest/src/chess/game.rs.html#98-105
                Some(self.turn().opposite())
            }
        })
    }

    fn flipped(&self) -> Self {
        let b = &self.board;

        let flip_rank = |r: chess::Rank| chess::Rank::from_index(ChessGame::BOARD_SIZE - 1 - r.to_index());
        let flip_file = |f: chess::File| f;
        let flip_square =
            |s: chess::Square| chess::Square::make_square(flip_rank(s.get_rank()), flip_file(s.get_file()));

        let pieces = b
            .combined()
            .into_iter()
            .map(|square| {
                (
                    flip_square(square),
                    b.piece_on(square).unwrap(),
                    !b.color_on(square).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let board = chess::Board::try_from(chess::BoardBuilder::setup(
            pieces.iter(),
            !b.side_to_move(),
            b.castle_rights(chess::Color::Black),
            b.castle_rights(chess::Color::White),
            b.en_passant().map(|square| flip_file(square.get_file())),
        ))
        .expect("unable to flip board");

        Self {
            board,
            fifty_rule_count: self.fifty_rule_count,
        }
    }
}

pub struct ChessGame {
    pos_history: Vec<ChessPosition>,
    seen_positions: HashMap<ChessPosition, u32>,
    repetition_detected: bool,
}

impl Game for ChessGame {
    type Position = ChessPosition;
    type Move = ChessMove;
    type Bitboard = ChessBitboard;
    const BOARD_SIZE: usize = 8;
    const MOVES_NUM: usize = 1880;
    const REPETITION_LIMIT: Option<usize> = Some(3);

    fn new() -> Self {
        Self::from_position(Self::Position::new())
    }

    fn from_position(pos: Self::Position) -> Self {
        Self {
            pos_history: vec![pos],
            seen_positions: HashMap::from([(pos, 1)]),
            repetition_detected: false,
        }
    }

    fn pos_history(&self) -> &[Self::Position] {
        &self.pos_history
    }

    fn status(&self) -> GameStatus {
        if self.repetition_detected {
            return GameStatus::Finished(None);
        }
        self.position().status()
    }

    fn play_single_turn(&mut self, next_move: Self::Move) {
        assert!(self.status().is_ongoing());
        let pos = self.position();
        let new_pos = pos.moved_position(next_move);
        self.pos_history.push(new_pos);

        let repeat = self.seen_positions.entry(new_pos).or_insert(0);
        *repeat += 1;
        if *repeat as usize >= Self::REPETITION_LIMIT.unwrap() {
            self.repetition_detected = true;
        }
    }
}

static NN_INDEX_TO_MOVE: std::sync::LazyLock<Vec<ChessMove>> = std::sync::LazyLock::new(|| {
    [
        "a1b1", "a1c1", "a1d1", "a1e1", "a1f1", "a1g1", "a1h1", "a1a2", "a1b2", "a1c2", "a1a3", "a1b3", "a1c3", "a1a4",
        "a1d4", "a1a5", "a1e5", "a1a6", "a1f6", "a1a7", "a1g7", "a1a8", "a1h8", "b1a1", "b1c1", "b1d1", "b1e1", "b1f1",
        "b1g1", "b1h1", "b1a2", "b1b2", "b1c2", "b1d2", "b1a3", "b1b3", "b1c3", "b1d3", "b1b4", "b1e4", "b1b5", "b1f5",
        "b1b6", "b1g6", "b1b7", "b1h7", "b1b8", "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1a2", "c1b2",
        "c1c2", "c1d2", "c1e2", "c1a3", "c1b3", "c1c3", "c1d3", "c1e3", "c1c4", "c1f4", "c1c5", "c1g5", "c1c6", "c1h6",
        "c1c7", "c1c8", "d1a1", "d1b1", "d1c1", "d1e1", "d1f1", "d1g1", "d1h1", "d1b2", "d1c2", "d1d2", "d1e2", "d1f2",
        "d1b3", "d1c3", "d1d3", "d1e3", "d1f3", "d1a4", "d1d4", "d1g4", "d1d5", "d1h5", "d1d6", "d1d7", "d1d8", "e1a1",
        "e1b1", "e1c1", "e1d1", "e1f1", "e1g1", "e1h1", "e1c2", "e1d2", "e1e2", "e1f2", "e1g2", "e1c3", "e1d3", "e1e3",
        "e1f3", "e1g3", "e1b4", "e1e4", "e1h4", "e1a5", "e1e5", "e1e6", "e1e7", "e1e8", "f1a1", "f1b1", "f1c1", "f1d1",
        "f1e1", "f1g1", "f1h1", "f1d2", "f1e2", "f1f2", "f1g2", "f1h2", "f1d3", "f1e3", "f1f3", "f1g3", "f1h3", "f1c4",
        "f1f4", "f1b5", "f1f5", "f1a6", "f1f6", "f1f7", "f1f8", "g1a1", "g1b1", "g1c1", "g1d1", "g1e1", "g1f1", "g1h1",
        "g1e2", "g1f2", "g1g2", "g1h2", "g1e3", "g1f3", "g1g3", "g1h3", "g1d4", "g1g4", "g1c5", "g1g5", "g1b6", "g1g6",
        "g1a7", "g1g7", "g1g8", "h1a1", "h1b1", "h1c1", "h1d1", "h1e1", "h1f1", "h1g1", "h1f2", "h1g2", "h1h2", "h1f3",
        "h1g3", "h1h3", "h1e4", "h1h4", "h1d5", "h1h5", "h1c6", "h1h6", "h1b7", "h1h7", "h1a8", "h1h8", "a2a1", "a2b1",
        "a2c1", "a2b2", "a2c2", "a2d2", "a2e2", "a2f2", "a2g2", "a2h2", "a2a3", "a2b3", "a2c3", "a2a4", "a2b4", "a2c4",
        "a2a5", "a2d5", "a2a6", "a2e6", "a2a7", "a2f7", "a2a8", "a2g8", "b2a1", "b2b1", "b2c1", "b2d1", "b2a2", "b2c2",
        "b2d2", "b2e2", "b2f2", "b2g2", "b2h2", "b2a3", "b2b3", "b2c3", "b2d3", "b2a4", "b2b4", "b2c4", "b2d4", "b2b5",
        "b2e5", "b2b6", "b2f6", "b2b7", "b2g7", "b2b8", "b2h8", "c2a1", "c2b1", "c2c1", "c2d1", "c2e1", "c2a2", "c2b2",
        "c2d2", "c2e2", "c2f2", "c2g2", "c2h2", "c2a3", "c2b3", "c2c3", "c2d3", "c2e3", "c2a4", "c2b4", "c2c4", "c2d4",
        "c2e4", "c2c5", "c2f5", "c2c6", "c2g6", "c2c7", "c2h7", "c2c8", "d2b1", "d2c1", "d2d1", "d2e1", "d2f1", "d2a2",
        "d2b2", "d2c2", "d2e2", "d2f2", "d2g2", "d2h2", "d2b3", "d2c3", "d2d3", "d2e3", "d2f3", "d2b4", "d2c4", "d2d4",
        "d2e4", "d2f4", "d2a5", "d2d5", "d2g5", "d2d6", "d2h6", "d2d7", "d2d8", "e2c1", "e2d1", "e2e1", "e2f1", "e2g1",
        "e2a2", "e2b2", "e2c2", "e2d2", "e2f2", "e2g2", "e2h2", "e2c3", "e2d3", "e2e3", "e2f3", "e2g3", "e2c4", "e2d4",
        "e2e4", "e2f4", "e2g4", "e2b5", "e2e5", "e2h5", "e2a6", "e2e6", "e2e7", "e2e8", "f2d1", "f2e1", "f2f1", "f2g1",
        "f2h1", "f2a2", "f2b2", "f2c2", "f2d2", "f2e2", "f2g2", "f2h2", "f2d3", "f2e3", "f2f3", "f2g3", "f2h3", "f2d4",
        "f2e4", "f2f4", "f2g4", "f2h4", "f2c5", "f2f5", "f2b6", "f2f6", "f2a7", "f2f7", "f2f8", "g2e1", "g2f1", "g2g1",
        "g2h1", "g2a2", "g2b2", "g2c2", "g2d2", "g2e2", "g2f2", "g2h2", "g2e3", "g2f3", "g2g3", "g2h3", "g2e4", "g2f4",
        "g2g4", "g2h4", "g2d5", "g2g5", "g2c6", "g2g6", "g2b7", "g2g7", "g2a8", "g2g8", "h2f1", "h2g1", "h2h1", "h2a2",
        "h2b2", "h2c2", "h2d2", "h2e2", "h2f2", "h2g2", "h2f3", "h2g3", "h2h3", "h2f4", "h2g4", "h2h4", "h2e5", "h2h5",
        "h2d6", "h2h6", "h2c7", "h2h7", "h2b8", "h2h8", "a3a1", "a3b1", "a3c1", "a3a2", "a3b2", "a3c2", "a3b3", "a3c3",
        "a3d3", "a3e3", "a3f3", "a3g3", "a3h3", "a3a4", "a3b4", "a3c4", "a3a5", "a3b5", "a3c5", "a3a6", "a3d6", "a3a7",
        "a3e7", "a3a8", "a3f8", "b3a1", "b3b1", "b3c1", "b3d1", "b3a2", "b3b2", "b3c2", "b3d2", "b3a3", "b3c3", "b3d3",
        "b3e3", "b3f3", "b3g3", "b3h3", "b3a4", "b3b4", "b3c4", "b3d4", "b3a5", "b3b5", "b3c5", "b3d5", "b3b6", "b3e6",
        "b3b7", "b3f7", "b3b8", "b3g8", "c3a1", "c3b1", "c3c1", "c3d1", "c3e1", "c3a2", "c3b2", "c3c2", "c3d2", "c3e2",
        "c3a3", "c3b3", "c3d3", "c3e3", "c3f3", "c3g3", "c3h3", "c3a4", "c3b4", "c3c4", "c3d4", "c3e4", "c3a5", "c3b5",
        "c3c5", "c3d5", "c3e5", "c3c6", "c3f6", "c3c7", "c3g7", "c3c8", "c3h8", "d3b1", "d3c1", "d3d1", "d3e1", "d3f1",
        "d3b2", "d3c2", "d3d2", "d3e2", "d3f2", "d3a3", "d3b3", "d3c3", "d3e3", "d3f3", "d3g3", "d3h3", "d3b4", "d3c4",
        "d3d4", "d3e4", "d3f4", "d3b5", "d3c5", "d3d5", "d3e5", "d3f5", "d3a6", "d3d6", "d3g6", "d3d7", "d3h7", "d3d8",
        "e3c1", "e3d1", "e3e1", "e3f1", "e3g1", "e3c2", "e3d2", "e3e2", "e3f2", "e3g2", "e3a3", "e3b3", "e3c3", "e3d3",
        "e3f3", "e3g3", "e3h3", "e3c4", "e3d4", "e3e4", "e3f4", "e3g4", "e3c5", "e3d5", "e3e5", "e3f5", "e3g5", "e3b6",
        "e3e6", "e3h6", "e3a7", "e3e7", "e3e8", "f3d1", "f3e1", "f3f1", "f3g1", "f3h1", "f3d2", "f3e2", "f3f2", "f3g2",
        "f3h2", "f3a3", "f3b3", "f3c3", "f3d3", "f3e3", "f3g3", "f3h3", "f3d4", "f3e4", "f3f4", "f3g4", "f3h4", "f3d5",
        "f3e5", "f3f5", "f3g5", "f3h5", "f3c6", "f3f6", "f3b7", "f3f7", "f3a8", "f3f8", "g3e1", "g3f1", "g3g1", "g3h1",
        "g3e2", "g3f2", "g3g2", "g3h2", "g3a3", "g3b3", "g3c3", "g3d3", "g3e3", "g3f3", "g3h3", "g3e4", "g3f4", "g3g4",
        "g3h4", "g3e5", "g3f5", "g3g5", "g3h5", "g3d6", "g3g6", "g3c7", "g3g7", "g3b8", "g3g8", "h3f1", "h3g1", "h3h1",
        "h3f2", "h3g2", "h3h2", "h3a3", "h3b3", "h3c3", "h3d3", "h3e3", "h3f3", "h3g3", "h3f4", "h3g4", "h3h4", "h3f5",
        "h3g5", "h3h5", "h3e6", "h3h6", "h3d7", "h3h7", "h3c8", "h3h8", "a4a1", "a4d1", "a4a2", "a4b2", "a4c2", "a4a3",
        "a4b3", "a4c3", "a4b4", "a4c4", "a4d4", "a4e4", "a4f4", "a4g4", "a4h4", "a4a5", "a4b5", "a4c5", "a4a6", "a4b6",
        "a4c6", "a4a7", "a4d7", "a4a8", "a4e8", "b4b1", "b4e1", "b4a2", "b4b2", "b4c2", "b4d2", "b4a3", "b4b3", "b4c3",
        "b4d3", "b4a4", "b4c4", "b4d4", "b4e4", "b4f4", "b4g4", "b4h4", "b4a5", "b4b5", "b4c5", "b4d5", "b4a6", "b4b6",
        "b4c6", "b4d6", "b4b7", "b4e7", "b4b8", "b4f8", "c4c1", "c4f1", "c4a2", "c4b2", "c4c2", "c4d2", "c4e2", "c4a3",
        "c4b3", "c4c3", "c4d3", "c4e3", "c4a4", "c4b4", "c4d4", "c4e4", "c4f4", "c4g4", "c4h4", "c4a5", "c4b5", "c4c5",
        "c4d5", "c4e5", "c4a6", "c4b6", "c4c6", "c4d6", "c4e6", "c4c7", "c4f7", "c4c8", "c4g8", "d4a1", "d4d1", "d4g1",
        "d4b2", "d4c2", "d4d2", "d4e2", "d4f2", "d4b3", "d4c3", "d4d3", "d4e3", "d4f3", "d4a4", "d4b4", "d4c4", "d4e4",
        "d4f4", "d4g4", "d4h4", "d4b5", "d4c5", "d4d5", "d4e5", "d4f5", "d4b6", "d4c6", "d4d6", "d4e6", "d4f6", "d4a7",
        "d4d7", "d4g7", "d4d8", "d4h8", "e4b1", "e4e1", "e4h1", "e4c2", "e4d2", "e4e2", "e4f2", "e4g2", "e4c3", "e4d3",
        "e4e3", "e4f3", "e4g3", "e4a4", "e4b4", "e4c4", "e4d4", "e4f4", "e4g4", "e4h4", "e4c5", "e4d5", "e4e5", "e4f5",
        "e4g5", "e4c6", "e4d6", "e4e6", "e4f6", "e4g6", "e4b7", "e4e7", "e4h7", "e4a8", "e4e8", "f4c1", "f4f1", "f4d2",
        "f4e2", "f4f2", "f4g2", "f4h2", "f4d3", "f4e3", "f4f3", "f4g3", "f4h3", "f4a4", "f4b4", "f4c4", "f4d4", "f4e4",
        "f4g4", "f4h4", "f4d5", "f4e5", "f4f5", "f4g5", "f4h5", "f4d6", "f4e6", "f4f6", "f4g6", "f4h6", "f4c7", "f4f7",
        "f4b8", "f4f8", "g4d1", "g4g1", "g4e2", "g4f2", "g4g2", "g4h2", "g4e3", "g4f3", "g4g3", "g4h3", "g4a4", "g4b4",
        "g4c4", "g4d4", "g4e4", "g4f4", "g4h4", "g4e5", "g4f5", "g4g5", "g4h5", "g4e6", "g4f6", "g4g6", "g4h6", "g4d7",
        "g4g7", "g4c8", "g4g8", "h4e1", "h4h1", "h4f2", "h4g2", "h4h2", "h4f3", "h4g3", "h4h3", "h4a4", "h4b4", "h4c4",
        "h4d4", "h4e4", "h4f4", "h4g4", "h4f5", "h4g5", "h4h5", "h4f6", "h4g6", "h4h6", "h4e7", "h4h7", "h4d8", "h4h8",
        "a5a1", "a5e1", "a5a2", "a5d2", "a5a3", "a5b3", "a5c3", "a5a4", "a5b4", "a5c4", "a5b5", "a5c5", "a5d5", "a5e5",
        "a5f5", "a5g5", "a5h5", "a5a6", "a5b6", "a5c6", "a5a7", "a5b7", "a5c7", "a5a8", "a5d8", "b5b1", "b5f1", "b5b2",
        "b5e2", "b5a3", "b5b3", "b5c3", "b5d3", "b5a4", "b5b4", "b5c4", "b5d4", "b5a5", "b5c5", "b5d5", "b5e5", "b5f5",
        "b5g5", "b5h5", "b5a6", "b5b6", "b5c6", "b5d6", "b5a7", "b5b7", "b5c7", "b5d7", "b5b8", "b5e8", "c5c1", "c5g1",
        "c5c2", "c5f2", "c5a3", "c5b3", "c5c3", "c5d3", "c5e3", "c5a4", "c5b4", "c5c4", "c5d4", "c5e4", "c5a5", "c5b5",
        "c5d5", "c5e5", "c5f5", "c5g5", "c5h5", "c5a6", "c5b6", "c5c6", "c5d6", "c5e6", "c5a7", "c5b7", "c5c7", "c5d7",
        "c5e7", "c5c8", "c5f8", "d5d1", "d5h1", "d5a2", "d5d2", "d5g2", "d5b3", "d5c3", "d5d3", "d5e3", "d5f3", "d5b4",
        "d5c4", "d5d4", "d5e4", "d5f4", "d5a5", "d5b5", "d5c5", "d5e5", "d5f5", "d5g5", "d5h5", "d5b6", "d5c6", "d5d6",
        "d5e6", "d5f6", "d5b7", "d5c7", "d5d7", "d5e7", "d5f7", "d5a8", "d5d8", "d5g8", "e5a1", "e5e1", "e5b2", "e5e2",
        "e5h2", "e5c3", "e5d3", "e5e3", "e5f3", "e5g3", "e5c4", "e5d4", "e5e4", "e5f4", "e5g4", "e5a5", "e5b5", "e5c5",
        "e5d5", "e5f5", "e5g5", "e5h5", "e5c6", "e5d6", "e5e6", "e5f6", "e5g6", "e5c7", "e5d7", "e5e7", "e5f7", "e5g7",
        "e5b8", "e5e8", "e5h8", "f5b1", "f5f1", "f5c2", "f5f2", "f5d3", "f5e3", "f5f3", "f5g3", "f5h3", "f5d4", "f5e4",
        "f5f4", "f5g4", "f5h4", "f5a5", "f5b5", "f5c5", "f5d5", "f5e5", "f5g5", "f5h5", "f5d6", "f5e6", "f5f6", "f5g6",
        "f5h6", "f5d7", "f5e7", "f5f7", "f5g7", "f5h7", "f5c8", "f5f8", "g5c1", "g5g1", "g5d2", "g5g2", "g5e3", "g5f3",
        "g5g3", "g5h3", "g5e4", "g5f4", "g5g4", "g5h4", "g5a5", "g5b5", "g5c5", "g5d5", "g5e5", "g5f5", "g5h5", "g5e6",
        "g5f6", "g5g6", "g5h6", "g5e7", "g5f7", "g5g7", "g5h7", "g5d8", "g5g8", "h5d1", "h5h1", "h5e2", "h5h2", "h5f3",
        "h5g3", "h5h3", "h5f4", "h5g4", "h5h4", "h5a5", "h5b5", "h5c5", "h5d5", "h5e5", "h5f5", "h5g5", "h5f6", "h5g6",
        "h5h6", "h5f7", "h5g7", "h5h7", "h5e8", "h5h8", "a6a1", "a6f1", "a6a2", "a6e2", "a6a3", "a6d3", "a6a4", "a6b4",
        "a6c4", "a6a5", "a6b5", "a6c5", "a6b6", "a6c6", "a6d6", "a6e6", "a6f6", "a6g6", "a6h6", "a6a7", "a6b7", "a6c7",
        "a6a8", "a6b8", "a6c8", "b6b1", "b6g1", "b6b2", "b6f2", "b6b3", "b6e3", "b6a4", "b6b4", "b6c4", "b6d4", "b6a5",
        "b6b5", "b6c5", "b6d5", "b6a6", "b6c6", "b6d6", "b6e6", "b6f6", "b6g6", "b6h6", "b6a7", "b6b7", "b6c7", "b6d7",
        "b6a8", "b6b8", "b6c8", "b6d8", "c6c1", "c6h1", "c6c2", "c6g2", "c6c3", "c6f3", "c6a4", "c6b4", "c6c4", "c6d4",
        "c6e4", "c6a5", "c6b5", "c6c5", "c6d5", "c6e5", "c6a6", "c6b6", "c6d6", "c6e6", "c6f6", "c6g6", "c6h6", "c6a7",
        "c6b7", "c6c7", "c6d7", "c6e7", "c6a8", "c6b8", "c6c8", "c6d8", "c6e8", "d6d1", "d6d2", "d6h2", "d6a3", "d6d3",
        "d6g3", "d6b4", "d6c4", "d6d4", "d6e4", "d6f4", "d6b5", "d6c5", "d6d5", "d6e5", "d6f5", "d6a6", "d6b6", "d6c6",
        "d6e6", "d6f6", "d6g6", "d6h6", "d6b7", "d6c7", "d6d7", "d6e7", "d6f7", "d6b8", "d6c8", "d6d8", "d6e8", "d6f8",
        "e6e1", "e6a2", "e6e2", "e6b3", "e6e3", "e6h3", "e6c4", "e6d4", "e6e4", "e6f4", "e6g4", "e6c5", "e6d5", "e6e5",
        "e6f5", "e6g5", "e6a6", "e6b6", "e6c6", "e6d6", "e6f6", "e6g6", "e6h6", "e6c7", "e6d7", "e6e7", "e6f7", "e6g7",
        "e6c8", "e6d8", "e6e8", "e6f8", "e6g8", "f6a1", "f6f1", "f6b2", "f6f2", "f6c3", "f6f3", "f6d4", "f6e4", "f6f4",
        "f6g4", "f6h4", "f6d5", "f6e5", "f6f5", "f6g5", "f6h5", "f6a6", "f6b6", "f6c6", "f6d6", "f6e6", "f6g6", "f6h6",
        "f6d7", "f6e7", "f6f7", "f6g7", "f6h7", "f6d8", "f6e8", "f6f8", "f6g8", "f6h8", "g6b1", "g6g1", "g6c2", "g6g2",
        "g6d3", "g6g3", "g6e4", "g6f4", "g6g4", "g6h4", "g6e5", "g6f5", "g6g5", "g6h5", "g6a6", "g6b6", "g6c6", "g6d6",
        "g6e6", "g6f6", "g6h6", "g6e7", "g6f7", "g6g7", "g6h7", "g6e8", "g6f8", "g6g8", "g6h8", "h6c1", "h6h1", "h6d2",
        "h6h2", "h6e3", "h6h3", "h6f4", "h6g4", "h6h4", "h6f5", "h6g5", "h6h5", "h6a6", "h6b6", "h6c6", "h6d6", "h6e6",
        "h6f6", "h6g6", "h6f7", "h6g7", "h6h7", "h6f8", "h6g8", "h6h8", "a7a1", "a7g1", "a7a2", "a7f2", "a7a3", "a7e3",
        "a7a4", "a7d4", "a7a5", "a7b5", "a7c5", "a7a6", "a7b6", "a7c6", "a7b7", "a7c7", "a7d7", "a7e7", "a7f7", "a7g7",
        "a7h7", "a7a8", "a7b8", "a7c8", "b7b1", "b7h1", "b7b2", "b7g2", "b7b3", "b7f3", "b7b4", "b7e4", "b7a5", "b7b5",
        "b7c5", "b7d5", "b7a6", "b7b6", "b7c6", "b7d6", "b7a7", "b7c7", "b7d7", "b7e7", "b7f7", "b7g7", "b7h7", "b7a8",
        "b7b8", "b7c8", "b7d8", "c7c1", "c7c2", "c7h2", "c7c3", "c7g3", "c7c4", "c7f4", "c7a5", "c7b5", "c7c5", "c7d5",
        "c7e5", "c7a6", "c7b6", "c7c6", "c7d6", "c7e6", "c7a7", "c7b7", "c7d7", "c7e7", "c7f7", "c7g7", "c7h7", "c7a8",
        "c7b8", "c7c8", "c7d8", "c7e8", "d7d1", "d7d2", "d7d3", "d7h3", "d7a4", "d7d4", "d7g4", "d7b5", "d7c5", "d7d5",
        "d7e5", "d7f5", "d7b6", "d7c6", "d7d6", "d7e6", "d7f6", "d7a7", "d7b7", "d7c7", "d7e7", "d7f7", "d7g7", "d7h7",
        "d7b8", "d7c8", "d7d8", "d7e8", "d7f8", "e7e1", "e7e2", "e7a3", "e7e3", "e7b4", "e7e4", "e7h4", "e7c5", "e7d5",
        "e7e5", "e7f5", "e7g5", "e7c6", "e7d6", "e7e6", "e7f6", "e7g6", "e7a7", "e7b7", "e7c7", "e7d7", "e7f7", "e7g7",
        "e7h7", "e7c8", "e7d8", "e7e8", "e7f8", "e7g8", "f7f1", "f7a2", "f7f2", "f7b3", "f7f3", "f7c4", "f7f4", "f7d5",
        "f7e5", "f7f5", "f7g5", "f7h5", "f7d6", "f7e6", "f7f6", "f7g6", "f7h6", "f7a7", "f7b7", "f7c7", "f7d7", "f7e7",
        "f7g7", "f7h7", "f7d8", "f7e8", "f7f8", "f7g8", "f7h8", "g7a1", "g7g1", "g7b2", "g7g2", "g7c3", "g7g3", "g7d4",
        "g7g4", "g7e5", "g7f5", "g7g5", "g7h5", "g7e6", "g7f6", "g7g6", "g7h6", "g7a7", "g7b7", "g7c7", "g7d7", "g7e7",
        "g7f7", "g7h7", "g7e8", "g7f8", "g7g8", "g7h8", "h7b1", "h7h1", "h7c2", "h7h2", "h7d3", "h7h3", "h7e4", "h7h4",
        "h7f5", "h7g5", "h7h5", "h7f6", "h7g6", "h7h6", "h7a7", "h7b7", "h7c7", "h7d7", "h7e7", "h7f7", "h7g7", "h7f8",
        "h7g8", "h7h8", "a8a1", "a8h1", "a8a2", "a8g2", "a8a3", "a8f3", "a8a4", "a8e4", "a8a5", "a8d5", "a8a6", "a8b6",
        "a8c6", "a8a7", "a8b7", "a8c7", "a8b8", "a8c8", "a8d8", "a8e8", "a8f8", "a8g8", "a8h8", "b8b1", "b8b2", "b8h2",
        "b8b3", "b8g3", "b8b4", "b8f4", "b8b5", "b8e5", "b8a6", "b8b6", "b8c6", "b8d6", "b8a7", "b8b7", "b8c7", "b8d7",
        "b8a8", "b8c8", "b8d8", "b8e8", "b8f8", "b8g8", "b8h8", "c8c1", "c8c2", "c8c3", "c8h3", "c8c4", "c8g4", "c8c5",
        "c8f5", "c8a6", "c8b6", "c8c6", "c8d6", "c8e6", "c8a7", "c8b7", "c8c7", "c8d7", "c8e7", "c8a8", "c8b8", "c8d8",
        "c8e8", "c8f8", "c8g8", "c8h8", "d8d1", "d8d2", "d8d3", "d8d4", "d8h4", "d8a5", "d8d5", "d8g5", "d8b6", "d8c6",
        "d8d6", "d8e6", "d8f6", "d8b7", "d8c7", "d8d7", "d8e7", "d8f7", "d8a8", "d8b8", "d8c8", "d8e8", "d8f8", "d8g8",
        "d8h8", "e8e1", "e8e2", "e8e3", "e8a4", "e8e4", "e8b5", "e8e5", "e8h5", "e8c6", "e8d6", "e8e6", "e8f6", "e8g6",
        "e8c7", "e8d7", "e8e7", "e8f7", "e8g7", "e8a8", "e8b8", "e8c8", "e8d8", "e8f8", "e8g8", "e8h8", "f8f1", "f8f2",
        "f8a3", "f8f3", "f8b4", "f8f4", "f8c5", "f8f5", "f8d6", "f8e6", "f8f6", "f8g6", "f8h6", "f8d7", "f8e7", "f8f7",
        "f8g7", "f8h7", "f8a8", "f8b8", "f8c8", "f8d8", "f8e8", "f8g8", "f8h8", "g8g1", "g8a2", "g8g2", "g8b3", "g8g3",
        "g8c4", "g8g4", "g8d5", "g8g5", "g8e6", "g8f6", "g8g6", "g8h6", "g8e7", "g8f7", "g8g7", "g8h7", "g8a8", "g8b8",
        "g8c8", "g8d8", "g8e8", "g8f8", "g8h8", "h8a1", "h8h1", "h8b2", "h8h2", "h8c3", "h8h3", "h8d4", "h8h4", "h8e5",
        "h8h5", "h8f6", "h8g6", "h8h6", "h8f7", "h8g7", "h8h7", "h8a8", "h8b8", "h8c8", "h8d8", "h8e8", "h8f8", "h8g8",
        "a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7b8q", "a7b8r", "a7b8b", "a7b8n", "b7a8q", "b7a8r", "b7a8b", "b7a8n",
        "b7b8q", "b7b8r", "b7b8b", "b7b8n", "b7c8q", "b7c8r", "b7c8b", "b7c8n", "c7b8q", "c7b8r", "c7b8b", "c7b8n",
        "c7c8q", "c7c8r", "c7c8b", "c7c8n", "c7d8q", "c7d8r", "c7d8b", "c7d8n", "d7c8q", "d7c8r", "d7c8b", "d7c8n",
        "d7d8q", "d7d8r", "d7d8b", "d7d8n", "d7e8q", "d7e8r", "d7e8b", "d7e8n", "e7d8q", "e7d8r", "e7d8b", "e7d8n",
        "e7e8q", "e7e8r", "e7e8b", "e7e8n", "e7f8q", "e7f8r", "e7f8b", "e7f8n", "f7e8q", "f7e8r", "f7e8b", "f7e8n",
        "f7f8q", "f7f8r", "f7f8b", "f7f8n", "f7g8q", "f7g8r", "f7g8b", "f7g8n", "g7f8q", "g7f8r", "g7f8b", "g7f8n",
        "g7g8q", "g7g8r", "g7g8b", "g7g8n", "g7h8q", "g7h8r", "g7h8b", "g7h8n", "h7g8q", "h7g8r", "h7g8b", "h7g8n",
        "h7h8q", "h7h8r", "h7h8b", "h7h8n",
    ]
    .into_iter()
    .map(|s| ChessMove::from_lan(s).unwrap())
    .collect::<Vec<_>>()
});

static MOVE_TO_NN_INDEX: std::sync::LazyLock<Vec<u16>> = std::sync::LazyLock::new(|| {
    let no_move = u16::MAX;
    let mut res = vec![no_move; 64 * 64 + 22 * 4];
    for (idx, m) in NN_INDEX_TO_MOVE.iter().enumerate() {
        assert_eq!(res[m.to_idx()], no_move);
        res[m.to_idx()] = idx as u16;
    }
    res
});

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::collections::HashSet;

    use crate::chess::{ChessGame, ChessMove, ChessPosition};
    use crate::game::player::{GamePlayer, PlayerRand};
    use crate::game::{Game, GameColor, GameStatus, Move, Position};

    #[test]
    fn simple_game_and_mate() {
        let mut pos = ChessPosition::new();

        let moves = vec![
            "e4", "e5", "d4", "exd4", "Qxd4", "Nc6", "Qa4", "a6", "Bg5", "h6", "Bc4", "Rb8", "Qb3", "Ra8", "Bxf7",
        ];
        for move_str in moves {
            assert!(pos.status().is_ongoing());
            let m = ChessMove::from_san(&pos, move_str).unwrap();
            pos = pos.moved_position(m);
        }
        assert_eq!(pos.status(), GameStatus::Finished(Some(GameColor::Player1)));
    }

    #[test]
    fn fifty_rule_count() {
        let mut pos = ChessPosition::new();

        #[rustfmt::skip]
        let moves = vec![
            "e4", "e5",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1",
        ];
        for move_str in moves {
            assert!(pos.status().is_ongoing());
            let m = ChessMove::from_san(&pos, move_str).unwrap();
            pos = pos.moved_position(m);
        }
        assert_eq!(pos.status(), GameStatus::Finished(None));
    }

    #[test]
    fn flip() {
        /* random FEN: */
        /* http://bernd.bplaced.net/fengenerator/fengenerator.html */
        for pos in [
            "7r/2B3n1/K5R1/3nPP2/P1k2Pp1/4p1p1/2p4P/8 w - - 0 1",
            "8/5pB1/1P4P1/1p3q2/BK1pP2P/pQ1pP3/7k/8 w - - 0 1",
            "2k5/2b1p1P1/1p5P/P1pnp3/QPK1b2p/8/5r2/8 b - - 0 1",
            "2b5/3N4/2B4p/1KP1R2r/n5p1/3N3P/k2B3b/q7 w - - 0 1",
            "8/2P1r3/4k1p1/4n1p1/3Rq3/P3Pp2/P4PP1/5K1N b - - 0 1",
            "1N6/5pP1/P1pK3P/P3p3/2R3p1/k3pQ2/6r1/6N1 b - - 0 1",
            "1B6/1r6/Q1ppKP2/qP1P1N1k/3p4/1pb5/7p/8 w - - 0 1",
            "3r4/1b2p3/7k/1P3R2/K4nrN/1N5P/n1Pp3P/8 b - - 0 1",
        ]
        .into_iter()
        .map(ChessPosition::from_fen)
        {
            assert_eq!(pos.turn().opposite(), pos.flipped().turn());
            assert_eq!(pos.flipped().flipped(), pos);
        }
    }

    #[test]
    fn flip_rand() {
        let mut rand = StdRng::seed_from_u64(0xb44d233429c09d16);

        let games_num = 100;
        for _ in 0..games_num {
            let mut player = PlayerRand::from_seed(rand.next_u64() ^ 0xe4655449311aee87);
            let mut game = ChessGame::new();

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

                let next_move = <_ as GamePlayer<ChessGame>>::next_move(&mut player, game.pos_history()).unwrap();
                game.play_single_turn(next_move);
            }
        }
    }
}
