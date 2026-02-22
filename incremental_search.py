#!/usr/bin/env python3
"""Incremental SAT search — add traces one by one using incremental solving.

Uses pysat's incremental interface so the solver retains learned clauses
between additions. Much faster than rebuilding from scratch each time.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List, Tuple

from pysat.solvers import Solver as SATSolver

from base_machine import TRANSITION_TABLE, SYMBOL_ENCODING
from sat_search import (
    VarManager,
    TransitionTable,
    add_trace,
    add_reversibility,
    extract_solution,
    print_table,
    save_table,
    ENTRY_L,
    ENTRY_R,
    MAX_MICRO,
    N_STATE_BITS,
)
from state_encoding import BinaryTransition, is_reversible, check_morita_reversibility

sys.stdout.reconfigure(line_buffering=True)


def run_incremental(with_reversibility: bool = False, with_r2l: bool = False,
                    save_path: str = "solution.json"):
    """Run incremental SAT search."""
    print("=" * 60)
    print("(32,2) Reversible UTM — Incremental SAT Search")
    print("=" * 60)
    print(f"Entry L: {ENTRY_L}")
    print(f"Entry R: {ENTRY_R}")
    print(f"Max micro-steps: {MAX_MICRO}")
    print(f"Reversibility: {with_reversibility}")
    print(f"R2L traces: {with_r2l}")
    print()

    vm = VarManager()
    table = TransitionTable(vm)
    all_clauses: List[List[int]] = []

    if with_reversibility:
        print("Adding reversibility constraints...")
        add_reversibility(vm, all_clauses, table)
        print(f"  {len(all_clauses)} clauses")

    # Create solver with initial clauses
    solver = SATSolver(name="cadical153", bootstrap_with=all_clauses)

    transitions = list(TRANSITION_TABLE.items())
    total_traces = len(transitions) * (2 if with_r2l else 1)
    trace_num = 0

    for idx, ((q, s), trans) in enumerate(transitions):
        bits_in = SYMBOL_ENCODING[s]
        bits_out = SYMBOL_ENCODING[trans.new_symbol]
        d = trans.direction
        d_str = "R" if d == 1 else "L"

        exit_head = 3 if d == 1 else -1
        exit_state = ENTRY_L[trans.new_state] if d == 1 else ENTRY_R[trans.new_state]

        # L2R trace
        new_clauses: List[List[int]] = []
        add_trace(vm, new_clauses, table, ENTRY_L[q], 0,
                  bits_in, bits_out, exit_head, exit_state)

        for c in new_clauses:
            solver.add_clause(c)
        trace_num += 1

        t0 = time.time()
        result = solver.solve()
        dt = time.time() - t0

        status = "SAT" if result else "UNSAT"
        print(f"  [{trace_num:2d}/{total_traces}] L2R Q={q} S={s} -> Q'={trans.new_state} S'={trans.new_symbol} {d_str}  "
              f"+{len(new_clauses):5d} cls  {status} ({dt:.1f}s)")

        if not result:
            print(f"\n  UNSAT after L2R trace for Q={q}, S={s}")
            return None

        if with_r2l:
            # R2L trace
            new_clauses = []
            add_trace(vm, new_clauses, table, ENTRY_R[q], 2,
                      bits_in, bits_out, exit_head, exit_state)

            for c in new_clauses:
                solver.add_clause(c)
            trace_num += 1

            t0 = time.time()
            result = solver.solve()
            dt = time.time() - t0

            status = "SAT" if result else "UNSAT"
            print(f"  [{trace_num:2d}/{total_traces}] R2L Q={q} S={s} -> Q'={trans.new_state} S'={trans.new_symbol} {d_str}  "
                  f"+{len(new_clauses):5d} cls  {status} ({dt:.1f}s)")

            if not result:
                print(f"\n  UNSAT after R2L trace for Q={q}, S={s}")
                return None

    print(f"\nAll {total_traces} traces: SAT!")
    print(f"Total: {vm.num_vars} vars")

    model = solver.get_model()
    solution = extract_solution(model, table)
    print_table(solution)

    print(f"\nLocally injective: {is_reversible(solution)}")
    print(f"Morita reversible: {check_morita_reversibility(solution)}")

    save_table(solution, save_path)
    return solution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reversibility", action="store_true")
    parser.add_argument("--r2l", action="store_true")
    parser.add_argument("--save", default="solution.json")
    args = parser.parse_args()

    run_incremental(
        with_reversibility=args.reversibility,
        with_r2l=args.r2l,
        save_path=args.save,
    )
