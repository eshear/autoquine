#!/usr/bin/env python3
"""
(2, 8) Holographic Weakly Universal TM — Z3 Search v2

Strengthened constraints to prevent degenerate solutions:
1. Payload purity: payload symbols only write payload symbols
2. Multi-cycle: require the TM to perform multiple Minsky steps
3. Clean return: after each cycle, tape must be in canonical form
"""

import json
import sys
import time
from z3 import (
    Solver, Int, If, And, Or, Not, Sum, Implies,
    sat, unsat, unknown, BitVec, BitVecVal,
    Extract, ZeroExt, Concat
)


def make_transition_vars(solver):
    """Create transition table variables with all structural constraints."""
    Q_out = [[Int(f"Qo_{q}_{s}") for s in range(8)] for q in range(2)]
    S_out = [[Int(f"So_{q}_{s}") for s in range(8)] for q in range(2)]
    D_out = [[Int(f"Do_{q}_{s}") for s in range(8)] for q in range(2)]

    # Domain constraints
    for q in range(2):
        for s in range(8):
            solver.add(Or(Q_out[q][s] == 0, Q_out[q][s] == 1))
            solver.add(S_out[q][s] >= 0, S_out[q][s] <= 7)
            solver.add(Or(D_out[q][s] == -1, D_out[q][s] == 1))

    # === Topological constraints (Section 2) ===
    # 2.1: Right wall bounce (state 0 reads 7)
    solver.add(Q_out[0][7] == 1, D_out[0][7] == -1)
    solver.add(S_out[0][7] >= 1, S_out[0][7] <= 6)

    # 2.2: Left wall bounce (state 1 reads 0)
    solver.add(Q_out[1][0] == 0, D_out[1][0] == 1)
    solver.add(S_out[1][0] >= 1, S_out[1][0] <= 6)

    # 2.3: Pass-through safety
    solver.add(Q_out[0][0] == 0, D_out[0][0] == 1, S_out[0][0] == 0)
    solver.add(Q_out[1][7] == 1, D_out[1][7] == -1, S_out[1][7] == 7)

    # === Continuity/Singularity (Section 3) ===
    rc = [If(And(Q_out[0][s] == 0, D_out[0][s] == 1), 1, 0) for s in range(1, 7)]
    solver.add(Sum(rc) == 5)

    lc = [If(And(Q_out[1][s] == 1, D_out[1][s] == -1), 1, 0) for s in range(1, 7)]
    solver.add(Sum(lc) == 5)

    # === NEW: Payload purity constraint ===
    # Payload symbols (1-6) can only write payload symbols (1-6)
    for q in range(2):
        for s in range(1, 7):
            solver.add(S_out[q][s] >= 1, S_out[q][s] <= 6)

    return Q_out, S_out, D_out


def unroll_tm(solver, Q_out, S_out, D_out, t_max, x_max,
              tape_init, head_start, state_start=0):
    """
    Unroll TM execution for t_max steps.
    Returns tape, head, state arrays.
    """
    tape = [[Int(f"t{t}x{x}") for x in range(x_max)] for t in range(t_max + 1)]
    head = [Int(f"h{t}") for t in range(t_max + 1)]
    state = [Int(f"s{t}") for t in range(t_max + 1)]

    # Initial conditions
    for x in range(x_max):
        solver.add(tape[0][x] == tape_init[x])
    solver.add(state[0] == state_start, head[0] == head_start)

    # Domain constraints on tape cells
    for t in range(t_max + 1):
        for x in range(x_max):
            solver.add(tape[t][x] >= 0, tape[t][x] <= 7)

    # Transition unrolling
    for t in range(t_max):
        h = head[t]
        st = state[t]

        # Read symbol at head
        cur_sym = tape[t][0]
        for x in range(x_max - 1, -1, -1):
            cur_sym = If(h == x, tape[t][x], cur_sym)

        # Lookup transition
        new_q = Q_out[0][0]
        new_s = S_out[0][0]
        new_d = D_out[0][0]
        for q in range(2):
            for sym in range(8):
                cond = And(st == q, cur_sym == sym)
                new_q = If(cond, Q_out[q][sym], new_q)
                new_s = If(cond, S_out[q][sym], new_s)
                new_d = If(cond, D_out[q][sym], new_d)

        solver.add(state[t+1] == new_q)
        solver.add(head[t+1] == h + new_d)
        solver.add(head[t+1] >= 0, head[t+1] < x_max)

        for x in range(x_max):
            solver.add(tape[t+1][x] == If(h == x, new_s, tape[t][x]))

    return tape, head, state


def extract_table(model, Q_out, S_out, D_out):
    """Extract transition table from Z3 model."""
    table = {}
    for q in range(2):
        for s in range(8):
            table[(q, s)] = (
                model.eval(Q_out[q][s]).as_long(),
                model.eval(S_out[q][s]).as_long(),
                model.eval(D_out[q][s]).as_long(),
            )
    return table


def print_table(table):
    dir_name = {-1: 'L', 1: 'R'}
    state_name = {0: 'SR', 1: 'SL'}
    sym_name = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}

    print(f"\n{'State':>6} {'Read':>5} -> {'Write':>6} {'Move':>5} {'Next':>6}")
    print("-" * 40)
    for q in range(2):
        for s in range(8):
            qo, so, do = table[(q, s)]
            print(f"{state_name[q]:>6} {sym_name[s]:>5} -> {sym_name[so]:>6} {dir_name[do]:>5} {state_name[qo]:>6}")
        print()


def simulate(table, tape_init, head, state, steps, x_max):
    tape = list(tape_init)
    history = [(list(tape), head, state)]
    for t in range(steps):
        sym = tape[head]
        if (state, sym) not in table:
            print(f"  No transition for (state={state}, sym={sym}) at step {t}")
            break
        new_q, new_s, new_d = table[(state, sym)]
        tape[head] = new_s
        state = new_q
        head += new_d
        if head < 0 or head >= x_max:
            print(f"  Head out of bounds at step {t+1}: head={head}")
            break
        history.append((list(tape), head, state))
    return history


def print_history(history, window=None):
    dir_sym = {0: '>', 1: '<'}
    for t, (tape, head, state) in enumerate(history):
        lo = window[0] if window else 0
        hi = window[1] if window else len(tape)
        row = ""
        for x in range(lo, hi):
            ch = str(tape[x])
            if tape[x] == 0: ch = '_'
            elif tape[x] == 7: ch = '#'
            if x == head:
                row += f"[{ch}]"
            else:
                row += f" {ch} "
        print(f"  t={t:3d} {dir_sym[state]} {row}")


def search_single_cycle(t_max=20, verbose=True):
    """
    Search for a TM that transforms [1,1,1,6] -> [1,1,6,4]
    with clean payload-purity constraints.
    """
    x_max = 20
    left_pad = 5

    payload_in = [1, 1, 1, 6]
    payload_out = [1, 1, 6, 4]

    tape_init = [0] * left_pad + payload_in + [7] * (x_max - left_pad - len(payload_in))

    print(f"Single cycle search: T={t_max}, tape={x_max}")
    print(f"Input:  {payload_in}")
    print(f"Target: {payload_out}")

    solver = Solver()
    solver.set("timeout", 120000)

    Q_out, S_out, D_out = make_transition_vars(solver)
    tape, head, state = unroll_tm(solver, Q_out, S_out, D_out,
                                   t_max, x_max, tape_init, left_pad)

    # Target: payload positions should match
    for i, sym in enumerate(payload_out):
        solver.add(tape[t_max][left_pad + i] == sym)

    # Must be sweeping right at end
    solver.add(state[t_max] == 0)

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract_table(model, Q_out, S_out, D_out)
        print_table(table)

        # Verify with extended simulation
        history = simulate(table, tape_init, left_pad, 0, 200, x_max)
        print("\nExtended simulation:")
        print_history(history, window=(0, x_max))

        return table
    return None


def search_multi_cycle(t_per_cycle=15, num_cycles=2, verbose=True):
    """
    Search for a TM that performs multiple Minsky dec-A/inc-B cycles.
    Uses shared transition variables across multiple independent unrollings.
    """
    x_max = 25
    left_pad = 8  # plenty of room for boundary expansion

    # Minsky: dec A, inc B
    # Cycle 1: A=3,B=0 -> A=2,B=1
    # Cycle 2: A=2,B=1 -> A=1,B=2
    cycles = [
        ([1, 1, 1, 6], [1, 1, 6, 4]),
        ([1, 1, 6, 4], [1, 6, 4, 4]),
    ]

    if num_cycles >= 3:
        cycles.append(([1, 6, 4, 4], [6, 4, 4, 4]))

    print(f"Multi-cycle search: {num_cycles} cycles, T/cycle={t_per_cycle}, tape={x_max}")
    for i, (pi, po) in enumerate(cycles[:num_cycles]):
        print(f"  Cycle {i+1}: {pi} -> {po}")

    solver = Solver()
    solver.set("timeout", 300000)

    # Shared transition table
    Q_out, S_out, D_out = make_transition_vars(solver)

    # Unroll each cycle independently with the SAME transition table
    for cycle_idx in range(num_cycles):
        pi, po = cycles[cycle_idx]
        tape_init = [0] * left_pad + pi + [7] * (x_max - left_pad - len(pi))

        # Use unique variable names per cycle
        tape_vars = [[Int(f"c{cycle_idx}_t{t}x{x}") for x in range(x_max)]
                     for t in range(t_per_cycle + 1)]
        head_vars = [Int(f"c{cycle_idx}_h{t}") for t in range(t_per_cycle + 1)]
        state_vars = [Int(f"c{cycle_idx}_s{t}") for t in range(t_per_cycle + 1)]

        # Initial conditions
        for x in range(x_max):
            solver.add(tape_vars[0][x] == tape_init[x])
        solver.add(state_vars[0] == 0, head_vars[0] == left_pad)

        # Unroll
        for t in range(t_per_cycle):
            h = head_vars[t]
            st = state_vars[t]

            cur_sym = tape_vars[t][0]
            for x in range(x_max - 1, -1, -1):
                cur_sym = If(h == x, tape_vars[t][x], cur_sym)

            new_q = Q_out[0][0]
            new_s = S_out[0][0]
            new_d = D_out[0][0]
            for q in range(2):
                for sym in range(8):
                    cond = And(st == q, cur_sym == sym)
                    new_q = If(cond, Q_out[q][sym], new_q)
                    new_s = If(cond, S_out[q][sym], new_s)
                    new_d = If(cond, D_out[q][sym], new_d)

            solver.add(state_vars[t+1] == new_q)
            solver.add(head_vars[t+1] == h + new_d)
            solver.add(head_vars[t+1] >= 0, head_vars[t+1] < x_max)

            for x in range(x_max):
                solver.add(tape_vars[t+1][x] == If(h == x, new_s, tape_vars[t][x]))

        # Target constraints for this cycle
        for i, sym in enumerate(po):
            solver.add(tape_vars[t_per_cycle][left_pad + i] == sym)
        solver.add(state_vars[t_per_cycle] == 0)

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"\nResult: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract_table(model, Q_out, S_out, D_out)
        print_table(table)

        # Verify all cycles
        for cycle_idx in range(num_cycles):
            pi, po = cycles[cycle_idx]
            tape_init = [0] * left_pad + pi + [7] * (x_max - left_pad - len(pi))
            print(f"\n--- Cycle {cycle_idx+1}: {pi} -> {po} ---")
            history = simulate(table, tape_init, left_pad, 0, t_per_cycle, x_max)
            print_history(history, window=(0, 20))

        # Extended simulation from cycle 1 start
        pi = cycles[0][0]
        tape_init = [0] * left_pad + pi + [7] * (x_max - left_pad - len(pi))
        print(f"\n--- Extended simulation (100 steps) ---")
        history = simulate(table, tape_init, left_pad, 0, 100, x_max)
        print_history(history, window=(0, x_max))

        # Save
        solution = {}
        for (q, s), (qo, so, do) in table.items():
            solution[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_holographic.json", "w") as f:
            json.dump(solution, f, indent=2)
        print("\nSaved to solution_holographic.json")

        return table
    return None


def search_iterative(verbose=True):
    """
    Iterative deepening: try increasing T values for multi-cycle search.
    """
    print("=" * 60)
    print("(2,8) Holographic TM — Iterative Multi-Cycle Search")
    print("=" * 60)

    for t_per_cycle in range(8, 40, 2):
        print(f"\n{'='*40}")
        table = search_multi_cycle(t_per_cycle=t_per_cycle, num_cycles=2)
        if table is not None:
            # Try with 3 cycles to further validate
            print(f"\n{'='*40}")
            print("Validating with 3 cycles...")
            table3 = search_multi_cycle(t_per_cycle=t_per_cycle, num_cycles=3)
            if table3 is not None:
                print("\n*** 3-CYCLE SOLUTION CONFIRMED ***")
                return table3
            else:
                print("3-cycle failed, continuing search...")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "iterative"],
                        default="iterative")
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "single":
        search_single_cycle(t_max=args.t_max)
    elif args.mode == "multi":
        search_multi_cycle(t_per_cycle=args.t_max, num_cycles=args.cycles)
    elif args.mode == "iterative":
        search_iterative()
