"""State encoding for the (32,2) reversible machine.

The 32 states decompose as F_2^3 ⊗ F_2^2:

  F_2^3 (logical_state): 8 values
    0..4  = macro-states corresponding to Q1..Q5 of the (5,5) machine
    5..7  = boundary/bounce reflector states used during 3-bit sweeps

  F_2^2 (spatial_gauge): 4 values
    phase ∈ {0, 1, 2}  = position within the 3-bit macro-block
    The 4th value encodes sweep direction:
      gauge = 0,1,2 with direction implied by transition
      gauge = 3 = "returning" (used during boundary bounce)

    More precisely, we encode:
      gauge = 2*phase + direction_bit
      where phase ∈ {0, 1} (we pack 3 phases into 2 bits + direction)

    Actually, cleaner encoding:
      gauge bits: [phase_hi, phase_lo] where:
        00 = phase 0 (reading bit 0 of macro-block)
        01 = phase 1 (reading bit 1 of macro-block)
        10 = phase 2 (reading bit 2 of macro-block)
        11 = bounce/return phase

Combined state = logical_state * 4 + spatial_gauge
  giving states 0..31.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

NUM_LOGICAL = 8   # F_2^3
NUM_GAUGE = 4     # F_2^2
NUM_STATES = NUM_LOGICAL * NUM_GAUGE  # 32
NUM_SYMBOLS = 2   # binary tape

# Logical state assignments
MACRO_STATES = list(range(5))       # 0..4 correspond to Q1..Q5
BOUNCE_STATES = list(range(5, 8))   # 5,6,7 are reflector states


def encode_state(logical: int, gauge: int) -> int:
    """Combine logical state and spatial gauge into a flat state index."""
    assert 0 <= logical < NUM_LOGICAL
    assert 0 <= gauge < NUM_GAUGE
    return logical * NUM_GAUGE + gauge


def decode_state(state: int) -> Tuple[int, int]:
    """Split a flat state index into (logical_state, spatial_gauge)."""
    assert 0 <= state < NUM_STATES
    return divmod(state, NUM_GAUGE)


@dataclass(frozen=True)
class BinaryTransition:
    """A transition for the (32,2) machine."""
    new_state: int     # 0..31
    new_symbol: int    # 0 or 1
    direction: int     # -1 (L) or +1 (R)

    def __post_init__(self):
        assert 0 <= self.new_state < NUM_STATES
        assert self.new_symbol in (0, 1)
        assert self.direction in (-1, 1)


def is_valid_transition_table(table: dict) -> bool:
    """Check that a transition table covers all (state, symbol) pairs."""
    for s in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            if (s, sym) not in table:
                return False
            t = table[(s, sym)]
            if not isinstance(t, BinaryTransition):
                return False
    return True


def is_reversible(table: dict) -> bool:
    """Check that the transition function is injective (reversible).

    For a reversible TM, the global map must be injective. A sufficient
    condition (for no-halt machines) is that the local transition function
    (state, symbol) -> (new_state, new_symbol, direction) is injective:
    no two distinct (state, symbol) inputs produce the same
    (new_state, new_symbol, direction) output.

    Note: this is actually stronger than necessary — true reversibility
    of the global map requires a more nuanced check (Morita's condition).
    We use this as a necessary condition and also check Morita's
    condition separately.
    """
    seen = set()
    for s in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            if (s, sym) not in table:
                return False
            t = table[(s, sym)]
            key = (t.new_state, t.new_symbol, t.direction)
            if key in seen:
                return False
            seen.add(key)
    return True


def check_morita_reversibility(table: dict) -> bool:
    """Check Morita's reversibility condition for a 1-tape TM.

    A deterministic TM is reversible (injective global function) iff
    it satisfies Morita's local conditions:

    For each state q' and direction d, the partial function
      f_{q',d} : symbol -> (state, symbol)
    defined by collecting all transitions that go to state q' moving
    direction d, i.e. {(s, sym) : table[(s, sym)] = (q', sym', d)}
    giving f_{q',d}(sym') = (s, sym), must be injective (i.e., each
    new_symbol value appears at most once among transitions arriving
    at q' from direction d).

    Equivalently: for each (new_state, direction) pair, no two transitions
    write the same new_symbol.
    """
    from collections import defaultdict

    # Group transitions by (new_state, direction)
    groups: dict[Tuple[int, int], list[Tuple[int, int, int]]] = defaultdict(list)
    for s in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            if (s, sym) not in table:
                return False
            t = table[(s, sym)]
            groups[(t.new_state, t.direction)].append((s, sym, t.new_symbol))

    for (ns, d), entries in groups.items():
        written_symbols = [e[2] for e in entries]
        if len(written_symbols) != len(set(written_symbols)):
            return False

    return True
