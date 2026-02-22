#!/usr/bin/env python3
"""Pure-SAT search for a (32,2) reversible UTM using pysat.

This reformulates the search as a Boolean satisfiability problem using
direct Boolean variables (no bitvec ITE trees).  The pysat library uses
CaDiCaL, one of the fastest SAT solvers available.

Key encoding decisions:
  - The transition table is encoded with one-hot variables for next_state
    (reduced to log-encoded bits) plus single bits for symbol/direction.
  - Entry states are fixed (not solver variables) to reduce search space.
  - Each trace unrolling step uses multiplexer circuits for table lookup.

State assignment (fixed):
  entry_L[q] = 2*q       for q=0..4  (states 0, 2, 4, 6, 8)
  entry_R[q] = 2*q + 1   for q=0..4  (states 1, 3, 5, 7, 9)
  States 10..31 are free intermediate/bounce states.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from typing import Dict, List, Optional, Tuple

from pysat.card import CardEnc
from pysat.solvers import Solver as SATSolver

from base_machine import (
    TRANSITION_TABLE as BASE_TABLE,
    SYMBOL_ENCODING,
    NUM_STATES as BASE_NUM_STATES,
    NUM_SYMBOLS as BASE_NUM_SYMBOLS,
)
from state_encoding import NUM_STATES, NUM_SYMBOLS, BinaryTransition, decode_state

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_MICRO = 8  # max micro-steps per macro-step (reduced from 10)
N_STATE_BITS = 5  # ceil(log2(32))

# Fixed entry states
ENTRY_L = [2 * q for q in range(BASE_NUM_STATES)]      # 0, 2, 4, 6, 8
ENTRY_R = [2 * q + 1 for q in range(BASE_NUM_STATES)]   # 1, 3, 5, 7, 9

# Head positions during a macro-step: -1, 0, 1, 2, 3
# We encode head position as a 3-bit signed value (range -4..3 covers -1..3)
N_HEAD_BITS = 3
HEAD_POSITIONS = list(range(-1, 4))

# ---------------------------------------------------------------------------
# Variable manager
# ---------------------------------------------------------------------------

class VarManager:
    """Allocate fresh SAT variables."""

    def __init__(self):
        self._next = 1

    def fresh(self) -> int:
        v = self._next
        self._next += 1
        return v

    def fresh_n(self, n: int) -> List[int]:
        return [self.fresh() for _ in range(n)]

    @property
    def num_vars(self) -> int:
        return self._next - 1


# ---------------------------------------------------------------------------
# Boolean encoding helpers
# ---------------------------------------------------------------------------

def int_to_bits(val: int, nbits: int) -> List[bool]:
    """Convert an integer to a list of bools (LSB first)."""
    if val < 0:
        val = val + (1 << nbits)  # two's complement
    return [(val >> i) & 1 == 1 for i in range(nbits)]


def bits_to_int(bits: List[bool], signed: bool = False) -> int:
    """Convert a list of bools (LSB first) to an integer."""
    val = sum(1 << i for i, b in enumerate(bits) if b)
    if signed and bits[-1]:  # MSB set = negative
        val -= 1 << len(bits)
    return val


def const_bits(val: int, nbits: int, vm: VarManager, clauses: list) -> List[int]:
    """Create SAT variables fixed to a constant value."""
    bits = int_to_bits(val, nbits)
    lits = vm.fresh_n(nbits)
    for i, b in enumerate(bits):
        clauses.append([lits[i] if b else -lits[i]])
    return lits


def equals_const(var_bits: List[int], val: int, nbits: int) -> int:
    """Return a literal that is true iff var_bits == val (as a conjunction).

    Actually, this returns a list of literals that must all be true.
    For use in implications/conditionals, we need an auxiliary variable.
    """
    target = int_to_bits(val, nbits)
    # For each bit, the literal is var_bits[i] if target[i] else -var_bits[i]
    return [var_bits[i] if target[i] else -var_bits[i] for i in range(nbits)]


def add_mux(vm: VarManager, clauses: list, sel_bits: List[int],
            data: Dict[int, int], result: int, default: Optional[int] = None):
    """Add a multiplexer: result = data[sel_bits] (single-bit output).

    For each possible value of sel_bits, if sel_bits==key, then result==data[key].
    data maps integer selector values to single SAT literals.
    """
    nbits = len(sel_bits)
    for key, data_lit in data.items():
        match_lits = equals_const(sel_bits, key, nbits)
        # If all match_lits are true, then result == data_lit
        # Clause: NOT(match_lit_1 AND ... AND match_lit_n AND result AND NOT data_lit)
        # = NOT(match) OR result == data_lit
        # This requires: match => (result <=> data_lit)
        # Which is: (NOT match OR NOT result OR data_lit) AND (NOT match OR result OR NOT data_lit)
        clauses.append([-l for l in match_lits] + [-result, data_lit])
        clauses.append([-l for l in match_lits] + [result, -data_lit])


def add_mux_multi(vm: VarManager, clauses: list, sel_bits: List[int],
                  data: Dict[int, List[int]], result_bits: List[int]):
    """Multi-bit multiplexer: result_bits = data[sel_bits].

    data maps integer selector values to lists of SAT literals (same length as result_bits).
    """
    nbits = len(sel_bits)
    nout = len(result_bits)
    for key, data_lits in data.items():
        assert len(data_lits) == nout
        match_lits = equals_const(sel_bits, key, nbits)
        for j in range(nout):
            clauses.append([-l for l in match_lits] + [-result_bits[j], data_lits[j]])
            clauses.append([-l for l in match_lits] + [result_bits[j], -data_lits[j]])


# ---------------------------------------------------------------------------
# Transition table variables
# ---------------------------------------------------------------------------

class TransitionTable:
    """SAT variables for the (32,2) transition table.

    For each (state, symbol) pair:
      ns_bits[s][a]: 5 Boolean vars (next state, LSB-first)
      na[s][a]: 1 Boolean var (next symbol: True=1)
      nd[s][a]: 1 Boolean var (next direction: True=R, False=L)
    """

    def __init__(self, vm: VarManager):
        self.ns_bits: Dict[Tuple[int, int], List[int]] = {}
        self.na: Dict[Tuple[int, int], int] = {}
        self.nd: Dict[Tuple[int, int], int] = {}

        for s in range(NUM_STATES):
            for a in range(NUM_SYMBOLS):
                self.ns_bits[(s, a)] = vm.fresh_n(N_STATE_BITS)
                self.na[(s, a)] = vm.fresh()
                self.nd[(s, a)] = vm.fresh()


# ---------------------------------------------------------------------------
# Lookup circuit: given symbolic state and symbol, get transition outputs
# ---------------------------------------------------------------------------

def lookup_transition(vm: VarManager, clauses: list, table: TransitionTable,
                      state_bits: List[int], symbol_lit: int
                      ) -> Tuple[List[int], int, int]:
    """Build a circuit that looks up the transition table.

    Given symbolic state (5 bits) and symbol (1 bit), returns:
      (new_state_bits, new_symbol_lit, new_dir_lit)

    Uses a 2-level mux: first mux on state, then on symbol.
    """
    # Combined selector: 6 bits = 5 (state) + 1 (symbol)
    sel_bits = state_bits + [symbol_lit]  # 6 bits total
    n_sel = len(sel_bits)

    new_state_bits = vm.fresh_n(N_STATE_BITS)
    new_symbol = vm.fresh()
    new_dir = vm.fresh()

    # Build data maps for the mux
    ns_data: Dict[int, List[int]] = {}
    na_data: Dict[int, int] = {}
    nd_data: Dict[int, int] = {}

    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            key = s + (a << N_STATE_BITS)  # combined key
            ns_data[key] = table.ns_bits[(s, a)]
            na_data[key] = table.na[(s, a)]
            nd_data[key] = table.nd[(s, a)]

    add_mux_multi(vm, clauses, sel_bits, ns_data, new_state_bits)
    add_mux(vm, clauses, sel_bits, na_data, new_symbol)
    add_mux(vm, clauses, sel_bits, nd_data, new_dir)

    return new_state_bits, new_symbol, new_dir


# ---------------------------------------------------------------------------
# Reversibility constraints
# ---------------------------------------------------------------------------

def add_reversibility(vm: VarManager, clauses: list, table: TransitionTable):
    """Morita reversibility: for each (new_state, direction), at most 2
    transitions target it, and if 2, they write different symbols.

    Encoded as: for all pairs (s1,a1) != (s2,a2), if they share the same
    (new_state, direction), they must write different symbols.
    """
    entries = [(s, a) for s in range(NUM_STATES) for a in range(NUM_SYMBOLS)]

    for i in range(len(entries)):
        s1, a1 = entries[i]
        for j in range(i + 1, len(entries)):
            s2, a2 = entries[j]

            # "same_state" = all 5 bits of ns match
            # "same_dir" = nd bits match
            # "same_sym" = na bits match
            # Constraint: NOT(same_state AND same_dir AND same_sym)

            # For each bit k: ns_bits[s1,a1][k] == ns_bits[s2,a2][k]
            # "all match AND dir match AND sym match" should be false.
            # This is equivalent to: at least one of them differs.

            # Direct encoding using auxiliary variables would be expensive
            # for 64*63/2 = 2016 pairs.  Instead, group by potential
            # new_state and use counting constraints.
            pass

    # Alternative: for each possible (new_state_val, direction_val) pair,
    # at most 2 transitions target it, and if 2, different symbols.
    # This is more efficient: 32 states × 2 dirs = 64 groups.
    for target_state in range(NUM_STATES):
        for target_dir in range(2):  # 0=L, 1=R
            # Collect indicator variables: trans (s,a) targets this (state, dir)
            indicators = []
            for s in range(NUM_STATES):
                for a in range(NUM_SYMBOLS):
                    # indicator = (ns_bits match target_state) AND (nd matches target_dir)
                    ind = vm.fresh()
                    ns_match = equals_const(table.ns_bits[(s, a)], target_state, N_STATE_BITS)
                    nd_lit = table.nd[(s, a)] if target_dir == 1 else -table.nd[(s, a)]

                    # ind => (ns_match AND nd_match)
                    # AND (ns_match AND nd_match) => ind
                    all_conds = ns_match + [nd_lit]
                    # ind => all_conds[k] for each k
                    for c in all_conds:
                        clauses.append([-ind, c])
                    # all_conds => ind
                    clauses.append([-c for c in all_conds] + [ind])

                    indicators.append((ind, s, a))

            # At most 2 transitions target this (state, dir)
            ind_lits = [ind for ind, _, _ in indicators]
            # Use at-most-2 cardinality constraint
            am2 = CardEnc.atmost(ind_lits, bound=2, top_id=vm.num_vars)
            for c in am2.clauses:
                clauses.append(c)
            # Update variable counter
            if am2.nv > vm.num_vars:
                vm._next = am2.nv + 1

            # If exactly 2 target this (state, dir), they must write different symbols
            for idx1 in range(len(indicators)):
                ind1, s1, a1 = indicators[idx1]
                for idx2 in range(idx1 + 1, len(indicators)):
                    ind2, s2, a2 = indicators[idx2]
                    # If both active: na[s1,a1] != na[s2,a2]
                    # (ind1 AND ind2) => (na1 XOR na2)
                    # = NOT(ind1 AND ind2 AND na1 AND na2)
                    #   AND NOT(ind1 AND ind2 AND NOT na1 AND NOT na2)
                    clauses.append([-ind1, -ind2, -table.na[(s1, a1)], -table.na[(s2, a2)]])
                    clauses.append([-ind1, -ind2, table.na[(s1, a1)], table.na[(s2, a2)]])

    print(f"  Reversibility: {len(clauses)} clauses so far")


# ---------------------------------------------------------------------------
# Trace unrolling for one macro-transition
# ---------------------------------------------------------------------------

def add_trace(vm: VarManager, clauses: list, table: TransitionTable,
              entry_state: int, init_head: int,
              bits_in: Tuple[int, ...], bits_out: Tuple[int, ...],
              exit_head: int, exit_state: int) -> None:
    """Add SAT clauses for one bounded trace.

    The trace tracks (state, head, tape, done) over MAX_MICRO steps.
    State: 5 bits, Head: 3 bits (signed), Tape: 5 bits (pos -1..3), Done: 1 bit.
    """
    K = MAX_MICRO

    # Trace variables at each time step
    st: List[List[int]] = []    # state bits
    hd: List[List[int]] = []    # head bits
    tp: List[Dict[int, int]] = []  # tape[pos] -> lit
    done: List[int] = []

    for t in range(K + 1):
        st.append(vm.fresh_n(N_STATE_BITS))
        hd.append(vm.fresh_n(N_HEAD_BITS))
        tp.append({p: vm.fresh() for p in HEAD_POSITIONS})
        done.append(vm.fresh())

    # --- Initial conditions ---
    entry_bits = int_to_bits(entry_state, N_STATE_BITS)
    for i in range(N_STATE_BITS):
        clauses.append([st[0][i] if entry_bits[i] else -st[0][i]])

    head_bits = int_to_bits(init_head, N_HEAD_BITS)
    for i in range(N_HEAD_BITS):
        clauses.append([hd[0][i] if head_bits[i] else -hd[0][i]])

    clauses.append([-done[0]])  # not done initially

    # Initial tape
    for i, p in enumerate([0, 1, 2]):
        clauses.append([tp[0][p] if bits_in[i] else -tp[0][p]])
    # tp[0][-1] and tp[0][3] are free (unconstrained)

    # --- Transition steps ---
    for t in range(K):
        # Read tape at head position
        sym_at_head = vm.fresh()
        # sym_at_head = tp[t][hd[t]]  -- mux over head position
        sym_data: Dict[int, int] = {}
        for p in HEAD_POSITIONS:
            p_enc = p if p >= 0 else p + (1 << N_HEAD_BITS)
            sym_data[p_enc] = tp[t][p]
        add_mux(vm, clauses, hd[t], sym_data, sym_at_head)

        # Lookup transition
        new_st, new_sym, new_dir = lookup_transition(
            vm, clauses, table, st[t], sym_at_head
        )

        # Compute new head: hd[t] + 1 if R, hd[t] - 1 if L
        # new_head = hd[t] + (1 if new_dir else -1)
        # We compute both options and mux
        hd_plus1 = vm.fresh_n(N_HEAD_BITS)
        hd_minus1 = vm.fresh_n(N_HEAD_BITS)
        add_increment(vm, clauses, hd[t], hd_plus1, +1, N_HEAD_BITS)
        add_increment(vm, clauses, hd[t], hd_minus1, -1, N_HEAD_BITS)

        new_head = vm.fresh_n(N_HEAD_BITS)
        # new_head = hd_plus1 if new_dir else hd_minus1
        for i in range(N_HEAD_BITS):
            # new_dir => new_head[i] == hd_plus1[i]
            clauses.append([-new_dir, -new_head[i], hd_plus1[i]])
            clauses.append([-new_dir, new_head[i], -hd_plus1[i]])
            # NOT new_dir => new_head[i] == hd_minus1[i]
            clauses.append([new_dir, -new_head[i], hd_minus1[i]])
            clauses.append([new_dir, new_head[i], -hd_minus1[i]])

        # Check if head exited: new_head == -1 or new_head == 3
        # -1 in 3-bit signed = 0b111 = 7, 3 = 0b011
        head_is_neg1 = vm.fresh()
        head_is_3 = vm.fresh()
        _add_eq_const(vm, clauses, new_head, -1, N_HEAD_BITS, head_is_neg1)
        _add_eq_const(vm, clauses, new_head, 3, N_HEAD_BITS, head_is_3)

        head_exited = vm.fresh()
        # head_exited = head_is_neg1 OR head_is_3
        clauses.append([-head_is_neg1, head_exited])
        clauses.append([-head_is_3, head_exited])
        clauses.append([head_is_neg1, head_is_3, -head_exited])

        # --- Update state, head, tape based on done flag ---

        # done[t+1] = done[t] OR head_exited  (when not already done)
        # Actually: done[t+1] = done[t] ? True : head_exited
        # = done[t] OR head_exited
        clauses.append([-done[t], done[t + 1]])      # done[t] => done[t+1]
        clauses.append([-head_exited, done[t + 1]])   # exited => done[t+1]
        # But also: if NOT done[t] AND NOT head_exited, then NOT done[t+1]
        # Wait, that's wrong. done[t+1] should be: done[t] OR (NOT done[t] AND head_exited)
        # = done[t] OR head_exited
        # But we also need: NOT done[t] AND NOT head_exited => NOT done[t+1]
        clauses.append([done[t], head_exited, -done[t + 1]])

        # State update: if done[t], freeze; else use new_st
        for i in range(N_STATE_BITS):
            # done[t] => st[t+1][i] == st[t][i]
            clauses.append([-done[t], -st[t + 1][i], st[t][i]])
            clauses.append([-done[t], st[t + 1][i], -st[t][i]])
            # NOT done[t] => st[t+1][i] == new_st[i]
            clauses.append([done[t], -st[t + 1][i], new_st[i]])
            clauses.append([done[t], st[t + 1][i], -new_st[i]])

        # Head update: if done[t], freeze; else use new_head
        for i in range(N_HEAD_BITS):
            clauses.append([-done[t], -hd[t + 1][i], hd[t][i]])
            clauses.append([-done[t], hd[t + 1][i], -hd[t][i]])
            clauses.append([done[t], -hd[t + 1][i], new_head[i]])
            clauses.append([done[t], hd[t + 1][i], -new_head[i]])

        # Tape update: if done[t], freeze all; else write new_sym at hd[t]
        for p in HEAD_POSITIONS:
            p_enc = p if p >= 0 else p + (1 << N_HEAD_BITS)
            head_at_p = vm.fresh()
            _add_eq_const(vm, clauses, hd[t], p, N_HEAD_BITS, head_at_p)

            # If done[t]: tp[t+1][p] = tp[t][p]
            clauses.append([-done[t], -tp[t + 1][p], tp[t][p]])
            clauses.append([-done[t], tp[t + 1][p], -tp[t][p]])

            # If NOT done[t] AND head_at_p: tp[t+1][p] = new_sym
            write_here = vm.fresh()
            clauses.append([-write_here, -done[t]])  # write_here => NOT done
            clauses.append([-write_here, head_at_p])  # write_here => head at p
            clauses.append([done[t], -head_at_p, write_here])  # NOT done AND head_at_p => write

            no_write = vm.fresh()
            # no_write = done[t] OR NOT head_at_p
            clauses.append([-done[t], no_write])
            clauses.append([head_at_p, no_write])
            clauses.append([done[t], -head_at_p, -no_write])

            # write_here => tp[t+1][p] == new_sym
            clauses.append([-write_here, -tp[t + 1][p], new_sym])
            clauses.append([-write_here, tp[t + 1][p], -new_sym])
            # no_write => tp[t+1][p] == tp[t][p] (but only if NOT done)
            # Actually, we already handled done[t] case above.
            # For NOT done AND NOT head_at_p: copy
            not_done_not_here = vm.fresh()
            clauses.append([done[t], head_at_p, not_done_not_here])
            clauses.append([-not_done_not_here, -done[t]])
            clauses.append([-not_done_not_here, -head_at_p])

            clauses.append([-not_done_not_here, -tp[t + 1][p], tp[t][p]])
            clauses.append([-not_done_not_here, tp[t + 1][p], -tp[t][p]])

    # --- Exit conditions ---
    clauses.append([done[K]])  # must be done

    # Head at exit position
    exit_bits = int_to_bits(exit_head, N_HEAD_BITS)
    for i in range(N_HEAD_BITS):
        clauses.append([hd[K][i] if exit_bits[i] else -hd[K][i]])

    # State at exit
    exit_st_bits = int_to_bits(exit_state, N_STATE_BITS)
    for i in range(N_STATE_BITS):
        clauses.append([st[K][i] if exit_st_bits[i] else -st[K][i]])

    # Tape at exit: positions 0,1,2 must match bits_out
    for idx, p in enumerate([0, 1, 2]):
        clauses.append([tp[K][p] if bits_out[idx] else -tp[K][p]])

    # Boundary cells unchanged
    # tp[K][-1] == tp[0][-1]
    clauses.append([-tp[K][-1], tp[0][-1]])
    clauses.append([tp[K][-1], -tp[0][-1]])
    # tp[K][3] == tp[0][3]
    clauses.append([-tp[K][3], tp[0][3]])
    clauses.append([tp[K][3], -tp[0][3]])


def add_increment(vm: VarManager, clauses: list,
                  a_bits: List[int], result_bits: List[int],
                  delta: int, nbits: int):
    """Constrain result_bits = a_bits + delta (mod 2^nbits).

    For delta = +1 or -1, uses a simple ripple-carry adder.
    """
    if delta == 1:
        # result = a + 1: flip bits up to the first 0
        carry = None
        for i in range(nbits):
            if i == 0:
                # result[0] = NOT a[0], carry = a[0]
                clauses.append([-result_bits[0], -a_bits[0]])
                clauses.append([result_bits[0], a_bits[0]])
                carry = a_bits[0]
            else:
                # result[i] = a[i] XOR carry
                # new_carry = a[i] AND carry
                xor_out = result_bits[i]
                new_carry = vm.fresh()
                _add_xor(clauses, a_bits[i], carry, xor_out)
                _add_and(clauses, a_bits[i], carry, new_carry)
                carry = new_carry
    elif delta == -1:
        # result = a - 1 = a + (2^n - 1): add all-ones
        # Equivalently: flip bits up to the first 1
        borrow = None
        for i in range(nbits):
            if i == 0:
                # result[0] = NOT a[0], borrow = NOT a[0]
                clauses.append([-result_bits[0], -a_bits[0]])
                clauses.append([result_bits[0], a_bits[0]])
                borrow = vm.fresh()
                clauses.append([-borrow, -a_bits[0]])
                clauses.append([borrow, a_bits[0]])
            else:
                # result[i] = a[i] XOR borrow
                # new_borrow = NOT a[i] AND borrow
                _add_xor(clauses, a_bits[i], borrow, result_bits[i])
                new_borrow = vm.fresh()
                not_a = vm.fresh()
                clauses.append([-not_a, -a_bits[i]])
                clauses.append([not_a, a_bits[i]])
                _add_and(clauses, not_a, borrow, new_borrow)
                borrow = new_borrow


def _add_xor(clauses: list, a: int, b: int, out: int):
    """out = a XOR b."""
    clauses.append([-a, -b, -out])
    clauses.append([-a, b, out])
    clauses.append([a, -b, out])
    clauses.append([a, b, -out])


def _add_and(clauses: list, a: int, b: int, out: int):
    """out = a AND b."""
    clauses.append([-a, -b, out])
    clauses.append([a, -out])
    clauses.append([b, -out])


def _add_eq_const(vm: VarManager, clauses: list,
                  var_bits: List[int], val: int, nbits: int,
                  indicator: int):
    """indicator <=> (var_bits == val)."""
    target = int_to_bits(val, nbits)
    match_lits = [var_bits[i] if target[i] else -var_bits[i] for i in range(nbits)]
    # indicator => all match_lits
    for ml in match_lits:
        clauses.append([-indicator, ml])
    # all match_lits => indicator
    clauses.append([-ml for ml in match_lits] + [indicator])


# ---------------------------------------------------------------------------
# Build the full problem
# ---------------------------------------------------------------------------

def build_problem():
    """Build the SAT problem and return (clauses, var_manager, table)."""
    vm = VarManager()
    clauses: List[List[int]] = []
    table = TransitionTable(vm)

    print(f"Transition table vars: {vm.num_vars}")

    # Reversibility
    print("Adding reversibility constraints...")
    add_reversibility(vm, clauses, table)
    print(f"  Total vars: {vm.num_vars}, clauses: {len(clauses)}")

    # Simulation traces
    print("Adding simulation traces...")
    n_traces = 0
    for (q, s), trans in BASE_TABLE.items():
        q_new = trans.new_state
        s_new = trans.new_symbol
        d = trans.direction

        bits_in = SYMBOL_ENCODING[s]
        bits_out = SYMBOL_ENCODING[s_new]

        # L2R trace
        if d == 1:
            exit_head, exit_state = 3, ENTRY_L[q_new]
        else:
            exit_head, exit_state = -1, ENTRY_R[q_new]

        add_trace(vm, clauses, table, ENTRY_L[q], 0,
                  bits_in, bits_out, exit_head, exit_state)
        n_traces += 1

        # R2L trace
        add_trace(vm, clauses, table, ENTRY_R[q], 2,
                  bits_in, bits_out, exit_head, exit_state)
        n_traces += 1

    print(f"  Added {n_traces} traces")
    print(f"  Total vars: {vm.num_vars}, clauses: {len(clauses)}")

    return clauses, vm, table


def solve(clauses, vm, table, timeout=600):
    """Solve the SAT problem."""
    print(f"\nSolving with CaDiCaL ({vm.num_vars} vars, {len(clauses)} clauses)...")

    solver = SATSolver(name="cadical153", bootstrap_with=clauses)

    start = time.time()
    result = solver.solve_limited(expect_interrupt=True)
    elapsed = time.time() - start

    if result is True:
        model = solver.get_model()
        print(f"SAT! ({elapsed:.1f}s)")
        return extract_solution(model, table)
    elif result is False:
        print(f"UNSAT ({elapsed:.1f}s)")
        return None
    else:
        print(f"UNKNOWN ({elapsed:.1f}s)")
        return None


def extract_solution(model, table):
    """Extract the transition table from a SAT model."""
    model_set = set(model)
    result = {}

    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            ns_val = 0
            for i, bit in enumerate(table.ns_bits[(s, a)]):
                if bit in model_set:
                    ns_val |= (1 << i)
            na_val = 1 if table.na[(s, a)] in model_set else 0
            nd_val = 1 if table.nd[(s, a)] in model_set else -1
            result[(s, a)] = BinaryTransition(ns_val, na_val, nd_val)

    return result


def print_table(tbl):
    """Pretty-print a transition table."""
    print(f"\n{'State':>5} {'Sym':>3} | {'NewState':>8} {'NewSym':>6} {'Dir':>3}")
    print("-" * 40)
    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            t = tbl[(s, a)]
            d_str = "R" if t.direction == 1 else "L"
            print(f"  {s:2d}     {a} |      {t.new_state:2d}      {t.new_symbol}     {d_str}")


def save_table(tbl, path):
    """Save to JSON."""
    data = {}
    for (s, a), t in tbl.items():
        data[f"{s},{a}"] = {"new_state": t.new_state, "new_symbol": t.new_symbol, "direction": t.direction}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Pure-SAT (32,2) reversible UTM search")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--save", type=str, default="solution.json")
    parser.add_argument("--no-reversibility", action="store_true",
                        help="Skip reversibility constraints (for debugging)")
    args = parser.parse_args()

    print("=" * 60)
    print("(32,2) Reversible UTM — Pure SAT Search (CaDiCaL)")
    print("=" * 60)
    print(f"States: {NUM_STATES}, Symbols: {NUM_SYMBOLS}")
    print(f"Entry L: {ENTRY_L}, Entry R: {ENTRY_R}")
    print(f"Max micro-steps: {MAX_MICRO}")
    print()

    clauses, vm, table = build_problem()
    result = solve(clauses, vm, table, timeout=args.timeout)

    if result:
        print_table(result)

        from state_encoding import is_reversible, check_morita_reversibility
        print(f"\nLocally injective: {is_reversible(result)}")
        print(f"Morita reversible: {check_morita_reversibility(result)}")

        save_table(result, args.save)


if __name__ == "__main__":
    main()
