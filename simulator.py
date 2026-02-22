"""Parallel simulator for verifying the (32,2) machine against the (5,5) machine.

Runs both machines step-by-step and checks that the (32,2) machine's
3-bit-encoded tape correctly tracks the (5,5) machine's tape after
each macro-step.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from base_machine import (
    TRANSITION_TABLE as BASE_TABLE,
    SYMBOL_ENCODING,
    BIT_DECODING,
    NUM_STATES as BASE_NUM_STATES,
    NUM_SYMBOLS as BASE_NUM_SYMBOLS,
)
from state_encoding import (
    NUM_STATES,
    NUM_SYMBOLS,
    NUM_GAUGE,
    MACRO_STATES,
    BinaryTransition,
    decode_state,
    encode_state,
)


class Tape:
    """Infinite tape backed by a defaultdict."""

    def __init__(self, default=0):
        self._cells: Dict[int, int] = defaultdict(lambda: default)

    def __getitem__(self, pos: int) -> int:
        return self._cells[pos]

    def __setitem__(self, pos: int, val: int):
        self._cells[pos] = val

    def to_dict(self) -> dict:
        return dict(self._cells)

    def segment(self, lo: int, hi: int) -> List[int]:
        """Return tape contents for positions [lo, hi)."""
        return [self._cells[p] for p in range(lo, hi)]


class BaseMachine:
    """Simulator for the (5,5) base machine."""

    def __init__(self):
        self.state = 0
        self.head = 0
        self.tape = Tape(default=0)
        self.steps = 0

    def step(self) -> bool:
        """Execute one step. Returns True if a transition was found."""
        sym = self.tape[self.head]
        key = (self.state, sym)
        if key not in BASE_TABLE:
            return False
        t = BASE_TABLE[key]
        self.tape[self.head] = t.new_symbol
        self.state = t.new_state
        self.head += t.direction
        self.steps += 1
        return True


class BinaryMachine:
    """Simulator for the (32,2) machine."""

    def __init__(self, table: Dict[Tuple[int, int], BinaryTransition]):
        self.table = table
        self.state = encode_state(0, 0)  # macro-state Q1, gauge 0
        self.head = 0
        self.tape = Tape(default=0)
        self.steps = 0

    def step(self) -> bool:
        """Execute one step. Returns True if a transition was found."""
        sym = self.tape[self.head]
        key = (self.state, sym)
        if key not in self.table:
            return False
        t = self.table[key]
        self.tape[self.head] = t.new_symbol
        self.state = t.new_state
        self.head += t.direction
        self.steps += 1
        return True

    def is_at_macro_boundary(self) -> bool:
        """Check if the machine is at the start of a new macro-step.

        This is true when the gauge is 0 (L2R entry) or 3 (R2L entry)
        and the logical state is a macro-state (0..4), meaning we've
        finished processing one block and are ready for the next.

        Actually, for simplicity: the machine is at a macro-boundary
        when the state's gauge is 0 and logical state is 0..4.
        """
        logical, gauge = decode_state(self.state)
        return logical in MACRO_STATES and gauge == 0


def encode_tape(base_tape: Tape, lo: int, hi: int) -> Tape:
    """Encode a (5,5) tape segment into a binary tape using 3-bit blocks.

    Position i on the base tape maps to positions 3*i, 3*i+1, 3*i+2
    on the binary tape.
    """
    binary_tape = Tape(default=0)
    for pos in range(lo, hi):
        sym = base_tape[pos]
        bits = SYMBOL_ENCODING[sym]
        for b in range(3):
            binary_tape[3 * pos + b] = bits[b]
    return binary_tape


def decode_tape_block(binary_tape: Tape, block_pos: int) -> Optional[int]:
    """Decode a 3-bit block from the binary tape back to a (5,5) symbol.

    block_pos is the base-tape position; binary positions are 3*block_pos + {0,1,2}.
    Returns None if the 3-bit pattern is not a valid symbol encoding.
    """
    bits = tuple(binary_tape[3 * block_pos + b] for b in range(3))
    return BIT_DECODING.get(bits)


def simulate_parallel(table: Dict[Tuple[int, int], BinaryTransition],
                      max_steps: int = 1000,
                      verbose: bool = True) -> bool:
    """Run both machines in parallel and verify correspondence.

    Returns True if the (32,2) machine correctly simulates the (5,5)
    machine for max_steps macro-steps.
    """
    base = BaseMachine()
    # Set up a simple test tape: some non-blank pattern
    # Use the standard test: tape = ...0 0 1 2 3 4 0 0...
    # centered around position 0
    test_symbols = [0, 1, 2, 3, 4, 1, 2, 0]
    for i, s in enumerate(test_symbols):
        base.tape[i] = s

    # Encode the base tape for the binary machine
    binary = BinaryMachine(table)
    tape_lo, tape_hi = -10, len(test_symbols) + 10
    for pos in range(tape_lo, tape_hi):
        sym = base.tape[pos]
        bits = SYMBOL_ENCODING[sym]
        for b in range(3):
            binary.tape[3 * pos + b] = bits[b]

    # The binary machine head starts at position 3*0 + 0 = 0
    binary.head = 3 * base.head

    macro_steps = 0
    errors = 0

    for macro_step in range(max_steps):
        # Step the base machine once
        if not base.step():
            if verbose:
                print(f"Base machine halted at step {macro_step}")
            break

        # Step the binary machine until it reaches a macro-boundary
        # (or we exceed a reasonable micro-step limit)
        micro_limit = 20  # a 3-bit sweep + bounce should take at most ~10 steps
        reached_boundary = False
        for _ in range(micro_limit):
            if not binary.step():
                if verbose:
                    print(f"Binary machine stuck at step {binary.steps}")
                return False
            if binary.is_at_macro_boundary():
                reached_boundary = True
                break

        if not reached_boundary:
            if verbose:
                print(f"Binary machine didn't reach macro-boundary "
                      f"after {micro_limit} micro-steps (macro-step {macro_step})")
            return False

        # Check correspondence
        bin_logical, bin_gauge = decode_state(binary.state)
        if bin_logical != base.state:
            if verbose:
                print(f"State mismatch at macro-step {macro_step}: "
                      f"base={base.state}, binary logical={bin_logical}")
            errors += 1
            if errors > 5:
                return False

        # Check tape correspondence around the head
        for pos in range(tape_lo, tape_hi):
            decoded = decode_tape_block(binary.tape, pos)
            if decoded is None:
                if verbose:
                    print(f"Invalid 3-bit code at block {pos} "
                          f"(macro-step {macro_step})")
                errors += 1
                break
            if decoded != base.tape[pos]:
                if verbose:
                    print(f"Tape mismatch at block {pos} "
                          f"(macro-step {macro_step}): "
                          f"base={base.tape[pos]}, decoded={decoded}")
                errors += 1
                break

        macro_steps += 1

    if errors == 0:
        if verbose:
            print(f"Verified {macro_steps} macro-steps successfully "
                  f"({binary.steps} binary steps)")
        return True
    else:
        if verbose:
            print(f"Found {errors} errors in {macro_steps} macro-steps")
        return False
