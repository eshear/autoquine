"""Tests for the (32,2) reversible UTM search infrastructure."""

import pytest

from base_machine import (
    TRANSITION_TABLE,
    SYMBOL_ENCODING,
    BIT_DECODING,
    UNUSED_CODES,
    validate_table,
    NUM_STATES as BASE_STATES,
    NUM_SYMBOLS as BASE_SYMBOLS,
)
from state_encoding import (
    NUM_STATES,
    NUM_SYMBOLS,
    NUM_LOGICAL,
    NUM_GAUGE,
    MACRO_STATES,
    BOUNCE_STATES,
    encode_state,
    decode_state,
    BinaryTransition,
    is_reversible,
    check_morita_reversibility,
)
from simulator import (
    Tape,
    BaseMachine,
    encode_tape,
    decode_tape_block,
)


# ---------------------------------------------------------------------------
# Base machine tests
# ---------------------------------------------------------------------------

class TestBaseMachine:
    def test_transition_table_complete(self):
        assert validate_table()

    def test_transition_table_size(self):
        assert len(TRANSITION_TABLE) == BASE_STATES * BASE_SYMBOLS

    def test_all_transitions_valid(self):
        for (s, sym), t in TRANSITION_TABLE.items():
            assert 0 <= s < BASE_STATES
            assert 0 <= sym < BASE_SYMBOLS
            assert 0 <= t.new_state < BASE_STATES
            assert 0 <= t.new_symbol < BASE_SYMBOLS
            assert t.direction in (-1, 1)


# ---------------------------------------------------------------------------
# Symbol encoding tests
# ---------------------------------------------------------------------------

class TestSymbolEncoding:
    def test_all_symbols_encoded(self):
        for s in range(BASE_SYMBOLS):
            assert s in SYMBOL_ENCODING
            assert len(SYMBOL_ENCODING[s]) == 3
            assert all(b in (0, 1) for b in SYMBOL_ENCODING[s])

    def test_encoding_injective(self):
        codes = list(SYMBOL_ENCODING.values())
        assert len(codes) == len(set(codes))

    def test_decoding_roundtrip(self):
        for s in range(BASE_SYMBOLS):
            bits = SYMBOL_ENCODING[s]
            assert BIT_DECODING[bits] == s

    def test_unused_codes(self):
        all_3bit = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        used = set(SYMBOL_ENCODING.values())
        unused = [c for c in all_3bit if c not in used]
        assert set(map(tuple, UNUSED_CODES)) == set(map(tuple, unused))


# ---------------------------------------------------------------------------
# State encoding tests
# ---------------------------------------------------------------------------

class TestStateEncoding:
    def test_state_count(self):
        assert NUM_STATES == 32
        assert NUM_SYMBOLS == 2
        assert NUM_LOGICAL == 8
        assert NUM_GAUGE == 4

    def test_encode_decode_roundtrip(self):
        for logical in range(NUM_LOGICAL):
            for gauge in range(NUM_GAUGE):
                state = encode_state(logical, gauge)
                assert 0 <= state < NUM_STATES
                l2, g2 = decode_state(state)
                assert l2 == logical
                assert g2 == gauge

    def test_all_states_reachable(self):
        seen = set()
        for logical in range(NUM_LOGICAL):
            for gauge in range(NUM_GAUGE):
                seen.add(encode_state(logical, gauge))
        assert seen == set(range(NUM_STATES))

    def test_macro_and_bounce_partition(self):
        assert set(MACRO_STATES) | set(BOUNCE_STATES) == set(range(NUM_LOGICAL))
        assert set(MACRO_STATES) & set(BOUNCE_STATES) == set()
        assert len(MACRO_STATES) == 5
        assert len(BOUNCE_STATES) == 3


# ---------------------------------------------------------------------------
# Reversibility check tests
# ---------------------------------------------------------------------------

class TestReversibility:
    def test_trivial_reversible_table(self):
        """A simple identity-like table that is trivially reversible."""
        table = {}
        for s in range(NUM_STATES):
            for a in range(NUM_SYMBOLS):
                # Each (s, a) maps to a unique (s', a', d) by construction
                new_s = (s + 1) % NUM_STATES if a == 0 else (s + 2) % NUM_STATES
                table[(s, a)] = BinaryTransition(new_s, a, 1 if s % 2 == 0 else -1)
        # This may or may not be reversible depending on collisions,
        # but we can at least test that the checker runs
        result = is_reversible(table)
        assert isinstance(result, bool)

    def test_irreversible_table(self):
        """A table where two transitions produce the same output."""
        table = {}
        for s in range(NUM_STATES):
            for a in range(NUM_SYMBOLS):
                # Everything goes to state 0, symbol 0, direction R
                table[(s, a)] = BinaryTransition(0, 0, 1)
        assert not is_reversible(table)
        assert not check_morita_reversibility(table)


# ---------------------------------------------------------------------------
# Tape and simulator tests
# ---------------------------------------------------------------------------

class TestTape:
    def test_default_value(self):
        t = Tape(default=0)
        assert t[100] == 0
        assert t[-50] == 0

    def test_read_write(self):
        t = Tape()
        t[5] = 1
        assert t[5] == 1
        assert t[4] == 0

    def test_segment(self):
        t = Tape()
        t[0] = 1
        t[1] = 0
        t[2] = 1
        assert t.segment(0, 3) == [1, 0, 1]


class TestBaseMachineSimulator:
    def test_single_step(self):
        m = BaseMachine()
        # Initial state 0, blank tape (symbol 0)
        # Transition (0, 0) -> (1, 2, R): state Q2, write s3, move R
        assert m.step()
        assert m.state == 1
        assert m.tape[0] == 2
        assert m.head == 1

    def test_multiple_steps(self):
        m = BaseMachine()
        for _ in range(10):
            assert m.step()
        assert m.steps == 10


class TestTapeEncoding:
    def test_encode_blank_tape(self):
        base_tape = Tape(default=0)
        binary_tape = encode_tape(base_tape, 0, 5)
        for pos in range(5):
            decoded = decode_tape_block(binary_tape, pos)
            assert decoded == 0

    def test_encode_roundtrip(self):
        base_tape = Tape(default=0)
        base_tape[0] = 1
        base_tape[1] = 4
        base_tape[2] = 2
        binary_tape = encode_tape(base_tape, 0, 5)
        for pos in range(5):
            decoded = decode_tape_block(binary_tape, pos)
            assert decoded == base_tape[pos]

    def test_all_symbols_roundtrip(self):
        base_tape = Tape(default=0)
        for s in range(BASE_SYMBOLS):
            base_tape[s] = s
        binary_tape = encode_tape(base_tape, 0, BASE_SYMBOLS)
        for s in range(BASE_SYMBOLS):
            assert decode_tape_block(binary_tape, s) == s


# ---------------------------------------------------------------------------
# Z3 solver smoke test
# ---------------------------------------------------------------------------

class TestSolverSmoke:
    def test_variables_created(self):
        from utm_search import make_transition_vars
        ns, na, nd = make_transition_vars()
        assert len(ns) == NUM_STATES * NUM_SYMBOLS
        assert len(na) == NUM_STATES * NUM_SYMBOLS
        assert len(nd) == NUM_STATES * NUM_SYMBOLS

    def test_entry_vars_created(self):
        from utm_search import make_entry_vars
        entry_L, entry_R = make_entry_vars()
        assert len(entry_L) == 5
        assert len(entry_R) == 5

    def test_solver_builds(self):
        """Smoke test: the solver builds without errors (may be slow)."""
        from utm_search import build_solver
        solver, ns, na, nd = build_solver()
        assert solver is not None
