#!/usr/bin/env python3
"""SAT-based search for a (32,2) reversible UTM simulating the (5,5) machine.

Approach: bounded trace unrolling.

The transition table has 32 * 2 = 64 entries, each specifying
(new_state, new_symbol, direction).  We search for a table satisfying:

1. Morita reversibility (injective global map on configurations).
2. Simulation correctness: for each of the 25 transitions of the (5,5)
   machine, a bounded sequence of micro-steps of the (32,2) machine
   correctly reads a 3-bit block, rewrites it, and exits in the correct
   state and direction.

The key insight: we do NOT prescribe the sweep strategy.  Instead, for each
macro-transition we unroll up to K micro-steps and let Z3 discover how the
machine reads, processes, and writes the 3-bit block.

Entry convention:
  - entry_L[q] : state when entering a block from the left (head at bit 0)
  - entry_R[q] : state when entering a block from the right (head at bit 2)
  These are Z3 variables — the solver chooses them.

For macro-transition (Q, S) -> (Q', S', D):
  bits_in  = SYMBOL_ENCODING[S]   (initial block contents)
  bits_out = SYMBOL_ENCODING[S']  (required final block contents)

  If arriving from the left:
    Start: state=entry_L[Q], head=0, tape=[bits_in]
    End:   head exits block (pos=-1 or pos=3)
           if D=R: head=3, state=entry_L[Q']
           if D=L: head=-1, state=entry_R[Q']
           tape=[bits_out]

  If arriving from the right:
    Start: state=entry_R[Q], head=2, tape=[bits_in]
    End:   similar, with sides swapped
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

from z3 import (
    And,
    BitVec,
    BitVecVal,
    Bool,
    BoolVal,
    If,
    Implies,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

from base_machine import (
    TRANSITION_TABLE as BASE_TABLE,
    SYMBOL_ENCODING,
    NUM_STATES as BASE_NUM_STATES,
    NUM_SYMBOLS as BASE_NUM_SYMBOLS,
)
from state_encoding import (
    NUM_STATES,
    NUM_SYMBOLS,
    decode_state,
    BinaryTransition,
)

# Max micro-steps per macro-step.  The head must traverse a 3-bit block
# (read + write + possible bounce), so ~10 steps suffices.
MAX_MICRO = 10

# Tape positions visible during a macro-step: the 3-bit block is at 0,1,2.
# The head may exit to -1 (left) or 3 (right).  We track positions -1..3.
TAPE_LO = -1
TAPE_HI = 4  # exclusive: positions -1, 0, 1, 2, 3
TAPE_POSITIONS = list(range(TAPE_LO, TAPE_HI))
NUM_TAPE_POS = len(TAPE_POSITIONS)


# ---------------------------------------------------------------------------
# Z3 variable creation
# ---------------------------------------------------------------------------

def make_transition_vars():
    """Create Z3 variables for the 64-entry transition table."""
    ns = {}  # next_state: BitVec(5)
    na = {}  # next_symbol: Bool
    nd = {}  # next_dir: Bool (False=L, True=R)
    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            ns[(s, a)] = BitVec(f"ns_{s}_{a}", 5)
            na[(s, a)] = Bool(f"na_{s}_{a}")
            nd[(s, a)] = Bool(f"nd_{s}_{a}")
    return ns, na, nd


def make_entry_vars():
    """Create Z3 variables for entry states.

    entry_L[q]: state when entering block from left  (q=0..4)
    entry_R[q]: state when entering block from right (q=0..4)
    """
    entry_L = [BitVec(f"eL_{q}", 5) for q in range(BASE_NUM_STATES)]
    entry_R = [BitVec(f"eR_{q}", 5) for q in range(BASE_NUM_STATES)]
    return entry_L, entry_R


# ---------------------------------------------------------------------------
# Transition table lookup as Z3 expression
# ---------------------------------------------------------------------------

def table_lookup_state(ns, state_expr, sym_expr):
    """Build a Z3 expression for next_state given symbolic state and symbol."""
    # Case-split over all 64 entries
    result = BitVecVal(0, 5)
    for s in range(NUM_STATES - 1, -1, -1):
        for a in range(NUM_SYMBOLS - 1, -1, -1):
            result = If(
                And(state_expr == s, sym_expr == BoolVal(a == 1)),
                ns[(s, a)],
                result,
            )
    return result


def table_lookup_symbol(na, state_expr, sym_expr):
    """Build a Z3 expression for next_symbol (Bool)."""
    result = BoolVal(False)
    for s in range(NUM_STATES - 1, -1, -1):
        for a in range(NUM_SYMBOLS - 1, -1, -1):
            result = If(
                And(state_expr == s, sym_expr == BoolVal(a == 1)),
                na[(s, a)],
                result,
            )
    return result


def table_lookup_dir(nd, state_expr, sym_expr):
    """Build a Z3 expression for direction (Bool: False=L, True=R)."""
    result = BoolVal(False)
    for s in range(NUM_STATES - 1, -1, -1):
        for a in range(NUM_SYMBOLS - 1, -1, -1):
            result = If(
                And(state_expr == s, sym_expr == BoolVal(a == 1)),
                nd[(s, a)],
                result,
            )
    return result


def tape_read(tape_vars, head_expr):
    """Read tape[head] where head is a symbolic expression.

    tape_vars: dict mapping position (int) -> Bool Z3 variable
    head_expr: BitVec(4) signed expression for head position (-1..3)
    Returns a Bool Z3 expression.
    """
    result = BoolVal(False)
    for p in TAPE_POSITIONS:
        result = If(head_expr == p, tape_vars[p], result)
    return result


# ---------------------------------------------------------------------------
# Bounded trace unrolling for one macro-transition
# ---------------------------------------------------------------------------

def add_macro_transition_trace(
    solver, ns, na, nd,
    entry_state_expr,   # Z3 BitVec(5): entry state
    init_head,          # int: initial head position (0 or 2)
    bits_in,            # tuple of 3 ints: initial block contents
    bits_out,           # tuple of 3 ints: required final block contents
    exit_head,          # int: required exit position (-1 or 3)
    exit_state_expr,    # Z3 BitVec(5): required exit state
    trace_prefix,       # str: prefix for trace variable names
):
    """Add constraints for one macro-transition's bounded trace.

    Unrolls MAX_MICRO micro-steps.  At each step the transition table
    governs the state/tape/head update.  We assert that at some step
    k <= MAX_MICRO, the head has exited the block with the correct state
    and the tape has the correct contents.
    """
    # Create trace variables
    # State at each time step: BitVec(5)
    st = [BitVec(f"{trace_prefix}_st{t}", 5) for t in range(MAX_MICRO + 1)]
    # Head position at each step: BitVec(4) signed (-1..3)
    # We use a 4-bit signed bitvec (range -8..7, sufficient for -1..3)
    hd = [BitVec(f"{trace_prefix}_hd{t}", 4) for t in range(MAX_MICRO + 1)]
    # Tape contents at each step: 5 Bools for positions -1..3
    tp = [{p: Bool(f"{trace_prefix}_tp{t}_{p}") for p in TAPE_POSITIONS}
          for t in range(MAX_MICRO + 1)]
    # "done" flag: once the head exits, computation stops
    done = [Bool(f"{trace_prefix}_done{t}") for t in range(MAX_MICRO + 1)]

    # --- Initial conditions ---
    solver.add(st[0] == entry_state_expr)
    solver.add(hd[0] == BitVecVal(init_head, 4))
    solver.add(done[0] == BoolVal(False))

    # Initial tape: block positions 0,1,2 have bits_in; positions -1,3 are
    # "boundary" cells from adjacent blocks.  We don't know their contents
    # a priori, so we leave them unconstrained (they're free variables).
    for i, p in enumerate([0, 1, 2]):
        solver.add(tp[0][p] == BoolVal(bits_in[i] == 1))
    # tp[0][-1] and tp[0][3] are free (adjacent block cells)

    # --- Transition steps ---
    for t in range(MAX_MICRO):
        # If already done, state/head/tape freeze
        sym_at_head = tape_read(tp[t], hd[t])
        new_state = table_lookup_state(ns, st[t], sym_at_head)
        new_sym = table_lookup_symbol(na, st[t], sym_at_head)
        new_dir = table_lookup_dir(nd, st[t], sym_at_head)

        # Head moves: +1 if dir=True(R), -1 if dir=False(L)
        new_head = If(new_dir,
                      hd[t] + BitVecVal(1, 4),
                      hd[t] + BitVecVal(-1, 4))  # -1 in 2's complement

        # When not done: update state, head, tape
        solver.add(If(done[t],
                       st[t + 1] == st[t],
                       st[t + 1] == new_state))
        solver.add(If(done[t],
                       hd[t + 1] == hd[t],
                       hd[t + 1] == new_head))

        # Tape update: write new_sym at head position, copy elsewhere
        for p in TAPE_POSITIONS:
            solver.add(If(done[t],
                           tp[t + 1][p] == tp[t][p],
                           If(hd[t] == p,
                              tp[t + 1][p] == new_sym,
                              tp[t + 1][p] == tp[t][p])))

        # Done flag: becomes true when head exits the block [0..2]
        # Head is at new_head after the step; if new_head is -1 or 3, we're done
        head_exited = Or(new_head == BitVecVal(-1, 4),
                         new_head == BitVecVal(3, 4))
        solver.add(If(done[t],
                       done[t + 1] == BoolVal(True),
                       done[t + 1] == head_exited))

    # --- Exit conditions ---
    # At the final step (or earlier), must be done
    solver.add(done[MAX_MICRO] == BoolVal(True))

    # Find the first step where done becomes true and check exit conditions.
    # We check at step MAX_MICRO: state and tape must be correct.
    # (Since state/head/tape freeze after done, checking at MAX_MICRO suffices.)
    solver.add(hd[MAX_MICRO] == BitVecVal(exit_head, 4))
    solver.add(st[MAX_MICRO] == exit_state_expr)

    # Tape at exit: positions 0,1,2 must contain bits_out
    for i, p in enumerate([0, 1, 2]):
        solver.add(tp[MAX_MICRO][p] == BoolVal(bits_out[i] == 1))

    # Boundary cells (-1, 3) should not have been modified.
    # (The head should only write within [0..2] before exiting.)
    # Actually, the head can land on -1 or 3 as the exit step, but the
    # done flag triggers before writing.  Let's just assert boundary
    # cells are unchanged.
    solver.add(tp[MAX_MICRO][-1] == tp[0][-1])
    solver.add(tp[MAX_MICRO][3] == tp[0][3])


# ---------------------------------------------------------------------------
# Constraint: Morita reversibility
# ---------------------------------------------------------------------------

def add_reversibility_constraints(solver, ns, na, nd):
    """Encode Morita's reversibility condition.

    For each (new_state, direction) pair, at most 2 incoming transitions,
    and if 2, they must write different symbols.
    """
    entries = [(s, a) for s in range(NUM_STATES) for a in range(NUM_SYMBOLS)]
    for i in range(len(entries)):
        s1, a1 = entries[i]
        for j in range(i + 1, len(entries)):
            s2, a2 = entries[j]
            same_target = And(
                ns[(s1, a1)] == ns[(s2, a2)],
                nd[(s1, a1)] == nd[(s2, a2)]
            )
            diff_symbol = na[(s1, a1)] != na[(s2, a2)]
            solver.add(Implies(same_target, diff_symbol))


# ---------------------------------------------------------------------------
# Constraint: valid ranges
# ---------------------------------------------------------------------------

def add_range_constraints(solver, ns):
    """Range constraints for next_state.

    With 5-bit bitvecs, values are already in 0..31 (= 2^5 - 1 = NUM_STATES - 1).
    No explicit constraints needed.
    """
    pass  # 5-bit BitVec naturally constrains to 0..31


# ---------------------------------------------------------------------------
# Constraint: entry state validity
# ---------------------------------------------------------------------------

def add_entry_constraints(solver, entry_L, entry_R):
    """Entry states must be distinct (no two macro-states share an entry).

    Range is already 0..31 from the 5-bit bitvec representation.
    """
    all_entries = entry_L + entry_R
    for i in range(len(all_entries)):
        for j in range(i + 1, len(all_entries)):
            solver.add(all_entries[i] != all_entries[j])


# ---------------------------------------------------------------------------
# Simulation constraints: one trace per macro-transition per entry direction
# ---------------------------------------------------------------------------

def add_simulation_constraints(solver, ns, na, nd, entry_L, entry_R):
    """For each (5,5) transition, add bounded trace constraints.

    Each of the 25 transitions needs TWO traces:
    - Entering the block from the left
    - Entering the block from the right
    Total: 50 bounded traces.
    """
    count = 0
    for (q, s), trans in BASE_TABLE.items():
        q_new = trans.new_state
        s_new = trans.new_symbol
        d = trans.direction  # +1 (R) or -1 (L)

        bits_in = SYMBOL_ENCODING[s]
        bits_out = SYMBOL_ENCODING[s_new]

        # --- Entering from the left (L2R) ---
        if d == 1:  # macro-direction R: exit right, next enters from left
            exit_head_l2r = 3
            exit_state_l2r = entry_L[q_new]
        else:       # macro-direction L: exit left, next enters from right
            exit_head_l2r = -1
            exit_state_l2r = entry_R[q_new]

        add_macro_transition_trace(
            solver, ns, na, nd,
            entry_state_expr=entry_L[q],
            init_head=0,
            bits_in=bits_in,
            bits_out=bits_out,
            exit_head=exit_head_l2r,
            exit_state_expr=exit_state_l2r,
            trace_prefix=f"L_{q}_{s}",
        )
        count += 1

        # --- Entering from the right (R2L) ---
        if d == 1:  # macro-direction R: exit right
            exit_head_r2l = 3
            exit_state_r2l = entry_L[q_new]
        else:       # macro-direction L: exit left
            exit_head_r2l = -1
            exit_state_r2l = entry_R[q_new]

        add_macro_transition_trace(
            solver, ns, na, nd,
            entry_state_expr=entry_R[q],
            init_head=2,
            bits_in=bits_in,
            bits_out=bits_out,
            exit_head=exit_head_r2l,
            exit_state_expr=exit_state_r2l,
            trace_prefix=f"R_{q}_{s}",
        )
        count += 1

    print(f"  Added {count} bounded traces (K={MAX_MICRO} micro-steps each)")


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def build_solver() -> Tuple[Solver, dict, dict, dict]:
    """Build the Z3 solver with all constraints."""
    solver = Solver()
    ns, na, nd = make_transition_vars()
    entry_L, entry_R = make_entry_vars()

    print("Adding range constraints...")
    add_range_constraints(solver, ns)

    print("Adding entry state constraints...")
    add_entry_constraints(solver, entry_L, entry_R)

    print("Adding reversibility constraints...")
    add_reversibility_constraints(solver, ns, na, nd)

    print("Adding simulation constraints (this may take a moment)...")
    add_simulation_constraints(solver, ns, na, nd, entry_L, entry_R)

    return solver, ns, na, nd


def extract_table(model, ns, na, nd) -> Dict[Tuple[int, int], BinaryTransition]:
    """Extract a concrete transition table from a Z3 model."""
    table = {}
    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            new_state = model.eval(ns[(s, a)]).as_long()
            new_sym = 1 if model.eval(na[(s, a)]) else 0
            new_dir = 1 if model.eval(nd[(s, a)]) else -1
            table[(s, a)] = BinaryTransition(new_state, new_sym, new_dir)
    return table


def print_table(table: Dict[Tuple[int, int], BinaryTransition]):
    """Pretty-print a transition table."""
    print(f"\n{'State':>5} {'Sym':>3} | {'NewState':>8} {'NewSym':>6} {'Dir':>3}")
    print("-" * 40)
    for s in range(NUM_STATES):
        for a in range(NUM_SYMBOLS):
            t = table[(s, a)]
            logical, gauge = decode_state(s)
            new_logical, new_gauge = decode_state(t.new_state)
            d_str = "R" if t.direction == 1 else "L"
            print(f"  {s:2d} ({logical},{gauge}) {a} | "
                  f"  {t.new_state:2d} ({new_logical},{new_gauge}) "
                  f"    {t.new_symbol}     {d_str}")


def save_table(table: Dict[Tuple[int, int], BinaryTransition], path: str):
    """Save a transition table to JSON."""
    data = {}
    for (s, a), t in table.items():
        data[f"{s},{a}"] = {
            "new_state": t.new_state,
            "new_symbol": t.new_symbol,
            "direction": t.direction,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path}")


def load_table(path: str) -> Dict[Tuple[int, int], BinaryTransition]:
    """Load a transition table from JSON."""
    with open(path) as f:
        data = json.load(f)
    table = {}
    for key, val in data.items():
        s, a = map(int, key.split(","))
        table[(s, a)] = BinaryTransition(val["new_state"], val["new_symbol"], val["direction"])
    return table


def verify_simulation(table: Dict[Tuple[int, int], BinaryTransition],
                      max_steps: int = 1000) -> bool:
    """Verify via simulation."""
    from simulator import simulate_parallel
    return simulate_parallel(table, max_steps=max_steps)


def main():
    parser = argparse.ArgumentParser(
        description="Search for a (32,2) reversible UTM via SAT/SMT"
    )
    parser.add_argument("--verify", type=str, default=None,
                        help="Verify a saved transition table (JSON path)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Solver timeout in seconds (default: 600)")
    parser.add_argument("--save", type=str, default="solution.json",
                        help="Path to save solution (default: solution.json)")
    args = parser.parse_args()

    if args.verify:
        table = load_table(args.verify)
        from state_encoding import is_reversible, check_morita_reversibility
        print(f"Loaded table from {args.verify}")
        print(f"  Locally injective: {is_reversible(table)}")
        print(f"  Morita reversible: {check_morita_reversibility(table)}")
        print(f"  Simulates (5,5):   {verify_simulation(table)}")
        return

    print("=" * 60)
    print("(32,2) Reversible UTM Search — Bounded Trace Unrolling")
    print("=" * 60)
    print(f"States: {NUM_STATES}, Symbols: {NUM_SYMBOLS}")
    print(f"Transition entries: {NUM_STATES * NUM_SYMBOLS}")
    print(f"Max micro-steps per macro-step: {MAX_MICRO}")
    print(f"Macro-transitions: {len(BASE_TABLE)} × 2 directions = {len(BASE_TABLE) * 2} traces")
    print()

    solver, ns, na, nd = build_solver()
    solver.set("timeout", args.timeout * 1000)

    print(f"\nSolving (timeout: {args.timeout}s)...")
    start = time.time()
    result = solver.check()
    elapsed = time.time() - start

    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract_table(model, ns, na, nd)
        print_table(table)

        from state_encoding import is_reversible, check_morita_reversibility
        print(f"\nLocally injective: {is_reversible(table)}")
        print(f"Morita reversible: {check_morita_reversibility(table)}")

        save_table(table, args.save)
        print(f"\nSolution saved to {args.save}")
    elif result == unsat:
        print("UNSAT — no solution with these constraints.")
    else:
        print("UNKNOWN (likely timeout). Try increasing --timeout or --max-micro.")


if __name__ == "__main__":
    main()
