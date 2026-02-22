"""The base (5,5) universal Turing machine.

This module defines the Rogozhin (5,5) UTM transition table that we are
trying to simulate with a (32,2) reversible machine.

States: q1..q5 (encoded 0..4)
Symbols: s1..s5 (encoded 0..4)
Directions: L = -1, R = +1

The transition table is from Rogozhin's 1996 paper "Small universal Turing
machines" (the variant proven universal by Kudlek and Rogozhin).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

L = -1
R = +1


class State(IntEnum):
    Q1 = 0
    Q2 = 1
    Q3 = 2
    Q4 = 3
    Q5 = 4


class Symbol(IntEnum):
    S1 = 0  # blank
    S2 = 1
    S3 = 2
    S4 = 3
    S5 = 4


NUM_STATES = 5
NUM_SYMBOLS = 5


@dataclass(frozen=True)
class Transition:
    """A single TM transition: (state, symbol) -> (new_state, new_symbol, direction)."""
    new_state: int
    new_symbol: int
    direction: int  # -1 (L) or +1 (R)


# Rogozhin's (5,5) UTM transition table.
# Format: (state, symbol) -> Transition(new_state, new_symbol, direction)
#
# This uses the Rogozhin 1996 encoding.  The exact table matters for the
# simulation constraints — if a different (5,5) variant is preferred, just
# swap this dict.
TRANSITION_TABLE: Dict[Tuple[int, int], Transition] = {
    # State Q1
    (0, 0): Transition(1, 2, R),   # q1, s1 -> q2, s3, R
    (0, 1): Transition(0, 2, L),   # q1, s2 -> q1, s3, L
    (0, 2): Transition(0, 4, L),   # q1, s3 -> q1, s5, L
    (0, 3): Transition(4, 0, R),   # q1, s4 -> q5, s1, R
    (0, 4): Transition(2, 0, L),   # q1, s5 -> q3, s1, L
    # State Q2
    (1, 0): Transition(0, 1, L),   # q2, s1 -> q1, s2, L
    (1, 1): Transition(1, 1, R),   # q2, s2 -> q2, s2, R
    (1, 2): Transition(1, 2, R),   # q2, s3 -> q2, s3, R
    (1, 3): Transition(3, 0, R),   # q2, s4 -> q4, s1, R
    (1, 4): Transition(0, 3, L),   # q2, s5 -> q1, s4, L
    # State Q3
    (2, 0): Transition(0, 0, L),   # q3, s1 -> q1, s1, L
    (2, 1): Transition(2, 1, L),   # q3, s2 -> q3, s2, L
    (2, 2): Transition(2, 2, L),   # q3, s3 -> q3, s3, L
    (2, 3): Transition(4, 0, R),   # q3, s4 -> q5, s1, R
    (2, 4): Transition(0, 3, R),   # q3, s5 -> q1, s4, R
    # State Q4
    (3, 0): Transition(0, 0, R),   # q4, s1 -> q1, s1, R
    (3, 1): Transition(3, 1, R),   # q4, s2 -> q4, s2, R
    (3, 2): Transition(3, 2, R),   # q4, s3 -> q4, s3, R
    (3, 3): Transition(4, 0, R),   # q4, s4 -> q5, s1, R
    (3, 4): Transition(0, 4, L),   # q4, s5 -> q1, s5, L
    # State Q5
    (4, 0): Transition(1, 3, R),   # q5, s1 -> q2, s4, R
    (4, 1): Transition(4, 1, R),   # q5, s2 -> q5, s2, R
    (4, 2): Transition(4, 2, R),   # q5, s3 -> q5, s3, R
    (4, 3): Transition(4, 3, R),   # q5, s4 -> q5, s4, R
    (4, 4): Transition(2, 4, L),   # q5, s5 -> q3, s5, L
}


def validate_table() -> bool:
    """Check that the transition table is complete for all (state, symbol) pairs."""
    for s in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            if (s, sym) not in TRANSITION_TABLE:
                return False
            t = TRANSITION_TABLE[(s, sym)]
            if not (0 <= t.new_state < NUM_STATES):
                return False
            if not (0 <= t.new_symbol < NUM_SYMBOLS):
                return False
            if t.direction not in (L, R):
                return False
    return True


# 3-bit encoding of the 5 symbols
# Symbols 0..4 map to 3-bit patterns; codes 5..7 are unused/padding.
SYMBOL_ENCODING: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # s1 (blank) -> 000
    1: (0, 0, 1),  # s2 -> 001
    2: (0, 1, 0),  # s3 -> 010
    3: (0, 1, 1),  # s4 -> 011
    4: (1, 0, 0),  # s5 -> 100
}

# Inverse mapping: 3-bit pattern -> symbol (only for valid codes)
BIT_DECODING: Dict[Tuple[int, int, int], int] = {v: k for k, v in SYMBOL_ENCODING.items()}

# Unused 3-bit codes (these should never appear on a properly-encoded tape)
UNUSED_CODES = [(1, 0, 1), (1, 1, 0), (1, 1, 1)]
