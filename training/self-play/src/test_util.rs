use std::cmp::Ordering;

use cattus::game::{Bitboard, Game as _, GameColor, Position};
use cattus::hex::{HexBitboard, HexPosition};
use cattus::ttt::{TttGame, TttPosition};
use itertools::Itertools;

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

pub fn hex_position_from_str<const BOARD_SIZE: usize>(s: &str) -> HexPosition<BOARD_SIZE> {
    let s = s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    assert_eq!(s.len(), BOARD_SIZE * BOARD_SIZE + 1, "unexpected string length");
    let lines = s
        .chars()
        .chunks(BOARD_SIZE)
        .into_iter()
        .map(|chunk| chunk.into_iter().collect_vec())
        .collect_vec();
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
                _ => panic!("unknown board char: {:?}", c),
            }
        }
    }

    let turn = match last_line[0] {
        'r' => GameColor::Player1,
        'b' => GameColor::Player2,
        unknown_turn_char => panic!("unknown turn char: {:?}", unknown_turn_char),
    };

    HexPosition::new_from_board(board_red, board_blue, turn)
}
