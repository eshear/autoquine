#!/usr/bin/env python3
"""
(2, 8) Holographic Weakly Universal Turing Machine — Z3 Synthesis

Searches for a 2-state, 8-symbol TM transition table that implements
a bounded spatial scanner (Minsky/Collatz simulator) on an open-system tape.

Symbols:
  0 = Left Infinite Boundary
  7 = Right Infinite Boundary
  1-6 = Computational Payload / Registers

States:
  0 = Sweep Right
  1 = Sweep Left

The boundary physics (wall bounces, pass-through) are hardcoded.
The solver only needs to find the 12 payload transitions (2 states × 6 symbols),
subject to continuity/singularity constraints.
"""

import json
import sys
import time
from z3 import (
    Solver, Int, If, And, Or, Sum, sat, unsat,
    IntVector, Array, IntSort, Store, Select
)


def build_solver(t_max, x_max=40, payload_in=None, payload_out=None,
                 head_start=16, verbose=True):
    """
    Build Z3 model for the (2,8) holographic TM.

    payload_in:  list of symbols for initial payload region (indices 16..24)
    payload_out: list of symbols expected in payload region at t=t_max
    """
    s = Solver()

    NUM_STATES = 2
    NUM_SYMBOLS = 8

    # === Transition table variables ===
    Q_out = [[Int(f"Q_out_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]
    S_out = [[Int(f"S_out_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]
    D_out = [[Int(f"D_out_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]

    # === Domain constraints ===
    for q in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            s.add(Or(Q_out[q][sym] == 0, Q_out[q][sym] == 1))
            s.add(S_out[q][sym] >= 0, S_out[q][sym] <= 7)
            s.add(Or(D_out[q][sym] == -1, D_out[q][sym] == 1))

    # === Section 2: Hardcoded Topological Constraints ===

    # 2.1: Right Wall Bounce — state 0 reads symbol 7
    s.add(Q_out[0][7] == 1)       # switch to sweep left
    s.add(D_out[0][7] == -1)      # bounce left
    s.add(S_out[0][7] >= 1, S_out[0][7] <= 6)  # write payload symbol

    # 2.2: Left Wall Bounce — state 1 reads symbol 0
    s.add(Q_out[1][0] == 0)       # switch to sweep right
    s.add(D_out[1][0] == 1)       # bounce right
    s.add(S_out[1][0] >= 1, S_out[1][0] <= 6)  # write payload symbol

    # 2.3: Pass-through Safety
    # State 0 reading boundary 0: pass through rightward
    s.add(Q_out[0][0] == 0)
    s.add(D_out[0][0] == 1)
    s.add(S_out[0][0] == 0)
    # State 1 reading boundary 7: pass through leftward
    s.add(Q_out[1][7] == 1)
    s.add(D_out[1][7] == -1)
    s.add(S_out[1][7] == 7)

    # === Section 3: Continuity / Singularity Constraints ===

    # For state 0 (sweep right), payload symbols 1-6:
    # Exactly 5 must maintain momentum (Q=0, D=+1), exactly 1 breaks
    right_continuous = []
    for sym in range(1, 7):
        is_cont = And(Q_out[0][sym] == 0, D_out[0][sym] == 1)
        right_continuous.append(If(is_cont, 1, 0))
    s.add(Sum(right_continuous) == 5)

    # For state 1 (sweep left), payload symbols 1-6:
    # Exactly 5 must maintain momentum (Q=1, D=-1), exactly 1 breaks
    left_continuous = []
    for sym in range(1, 7):
        is_cont = And(Q_out[1][sym] == 1, D_out[1][sym] == -1)
        left_continuous.append(If(is_cont, 1, 0))
    s.add(Sum(left_continuous) == 5)

    # === Section 4 & 5: Tape Simulation (Bounded Model Checking) ===

    # Tape as 2D array of Z3 ints: tape[t][x]
    tape = [[Int(f"tape_{t}_{x}") for x in range(x_max)] for t in range(t_max + 1)]
    head = [Int(f"head_{t}") for t in range(t_max + 1)]
    state = [Int(f"state_{t}") for t in range(t_max + 1)]

    # Initial tape setup
    if payload_in is None:
        # Default: Register A=3, Register B=0 encoded as [1,1,1,4,4] + padding
        payload_in = [1, 1, 1, 4, 4, 2, 2, 3, 3]

    for x in range(x_max):
        if x < 16:
            s.add(tape[0][x] == 0)
        elif x < 16 + len(payload_in):
            s.add(tape[0][x] == payload_in[x - 16])
        elif x < 25:
            # Fill remaining payload region with a neutral symbol
            s.add(tape[0][x] == 3)
        else:
            s.add(tape[0][x] == 7)

    s.add(state[0] == 0)
    s.add(head[0] == head_start)

    # Transition unrolling
    for t in range(t_max):
        h = head[t]
        st = state[t]

        # Read current symbol — we need to case-split on head position
        # For each possible head position, constrain the transition
        for x in range(x_max):
            sym = tape[t][x]

            # Build transition for each (state, symbol) pair
            for q in range(NUM_STATES):
                for sv in range(NUM_SYMBOLS):
                    cond = And(h == x, st == q, sym == sv)
                    s.add(If(cond,
                        And(
                            state[t+1] == Q_out[q][sv],
                            tape[t+1][x] == S_out[q][sv],
                            head[t+1] == x + D_out[q][sv]
                        ),
                        True
                    ))

        # Cells not under head stay unchanged
        for x2 in range(x_max):
            s.add(If(h != x2, tape[t+1][x2] == tape[t][x2], True))

        # Head stays in bounds
        s.add(head[t+1] >= 0, head[t+1] < x_max)

    # === Target assertion ===
    if payload_out is not None:
        for i, sym in enumerate(payload_out):
            s.add(tape[t_max][16 + i] == sym)

    return s, {
        'Q_out': Q_out, 'S_out': S_out, 'D_out': D_out,
        'tape': tape, 'head': head, 'state': state,
        't_max': t_max, 'x_max': x_max
    }


def build_solver_optimized(t_max, x_max=40, payload_in=None, payload_out=None,
                           head_start=16, target_state=None, target_head_region=None,
                           verbose=True):
    """
    Optimized version using fewer implications.
    Instead of case-splitting on all (x, q, s) combos, we use
    a chained If-Then-Else to look up the transition.
    """
    s = Solver()
    s.set("timeout", 300000)  # 5 min timeout per check

    NUM_STATES = 2
    NUM_SYMBOLS = 8

    # === Transition table variables ===
    Q_out = [[Int(f"Qo_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]
    S_out = [[Int(f"So_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]
    D_out = [[Int(f"Do_{q}_{sym}") for sym in range(NUM_SYMBOLS)] for q in range(NUM_STATES)]

    # Domain constraints
    for q in range(NUM_STATES):
        for sym in range(NUM_SYMBOLS):
            s.add(Or(Q_out[q][sym] == 0, Q_out[q][sym] == 1))
            s.add(S_out[q][sym] >= 0, S_out[q][sym] <= 7)
            s.add(Or(D_out[q][sym] == -1, D_out[q][sym] == 1))

    # === Hardcoded Topological Constraints ===
    # 2.1: Right Wall Bounce
    s.add(Q_out[0][7] == 1, D_out[0][7] == -1)
    s.add(S_out[0][7] >= 1, S_out[0][7] <= 6)

    # 2.2: Left Wall Bounce
    s.add(Q_out[1][0] == 0, D_out[1][0] == 1)
    s.add(S_out[1][0] >= 1, S_out[1][0] <= 6)

    # 2.3: Pass-through Safety
    s.add(Q_out[0][0] == 0, D_out[0][0] == 1, S_out[0][0] == 0)
    s.add(Q_out[1][7] == 1, D_out[1][7] == -1, S_out[1][7] == 7)

    # === Continuity / Singularity ===
    right_cont = [If(And(Q_out[0][sym] == 0, D_out[0][sym] == 1), 1, 0) for sym in range(1, 7)]
    s.add(Sum(right_cont) == 5)

    left_cont = [If(And(Q_out[1][sym] == 1, D_out[1][sym] == -1), 1, 0) for sym in range(1, 7)]
    s.add(Sum(left_cont) == 5)

    # === Tape Simulation ===
    if payload_in is None:
        payload_in = [1, 1, 1, 4, 4, 2, 2, 3, 3]

    # Use concrete integers for tape (flatten to 1D)
    tape = [[Int(f"t{t}x{x}") for x in range(x_max)] for t in range(t_max + 1)]
    head = [Int(f"h{t}") for t in range(t_max + 1)]
    state = [Int(f"s{t}") for t in range(t_max + 1)]

    # Initial conditions
    for x in range(x_max):
        if x < 16:
            s.add(tape[0][x] == 0)
        elif x < 16 + len(payload_in):
            s.add(tape[0][x] == payload_in[x - 16])
        elif x < 25:
            s.add(tape[0][x] == 3)
        else:
            s.add(tape[0][x] == 7)

    s.add(state[0] == 0, head[0] == head_start)

    # Unroll transitions using chained ITE for symbol lookup
    for t in range(t_max):
        h = head[t]
        st = state[t]

        # Read symbol at head position via chained ITE
        cur_sym = tape[t][0]  # fallback
        for x in range(x_max - 1, -1, -1):
            cur_sym = If(h == x, tape[t][x], cur_sym)

        # Lookup transition via chained ITE over (state, symbol)
        new_q = Q_out[0][0]  # fallback
        new_s = S_out[0][0]
        new_d = D_out[0][0]
        for q in range(NUM_STATES):
            for sym in range(NUM_SYMBOLS):
                cond = And(st == q, cur_sym == sym)
                new_q = If(cond, Q_out[q][sym], new_q)
                new_s = If(cond, S_out[q][sym], new_s)
                new_d = If(cond, D_out[q][sym], new_d)

        # Apply transition
        s.add(state[t+1] == new_q)
        s.add(head[t+1] == h + new_d)
        s.add(head[t+1] >= 0, head[t+1] < x_max)

        # Update tape: cell under head gets new symbol, rest unchanged
        for x in range(x_max):
            s.add(tape[t+1][x] == If(h == x, new_s, tape[t][x]))

    # === Target assertions ===
    if payload_out is not None:
        for i, sym in enumerate(payload_out):
            s.add(tape[t_max][16 + i] == sym)

    if target_state is not None:
        s.add(state[t_max] == target_state)

    if target_head_region is not None:
        lo, hi = target_head_region
        s.add(head[t_max] >= lo, head[t_max] <= hi)

    return s, {
        'Q_out': Q_out, 'S_out': S_out, 'D_out': D_out,
        'tape': tape, 'head': head, 'state': state,
        't_max': t_max, 'x_max': x_max
    }


def extract_transition_table(model, Q_out, S_out, D_out):
    """Extract solved transition table from Z3 model."""
    table = {}
    for q in range(2):
        for sym in range(8):
            qo = model.eval(Q_out[q][sym]).as_long()
            so = model.eval(S_out[q][sym]).as_long()
            do = model.eval(D_out[q][sym]).as_long()
            table[(q, sym)] = (qo, so, do)
    return table


def print_table(table):
    """Pretty-print transition table."""
    dir_name = {-1: 'L', 1: 'R'}
    state_name = {0: 'SR', 1: 'SL'}
    sym_name = {0: '⌐', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '¬'}

    print(f"\n{'State':>6} {'Read':>5} -> {'Write':>6} {'Move':>5} {'Next':>6}")
    print("-" * 40)
    for q in range(2):
        for sym in range(8):
            qo, so, do = table[(q, sym)]
            print(f"{state_name[q]:>6} {sym_name[sym]:>5} -> {sym_name[so]:>6} {dir_name[do]:>5} {state_name[qo]:>6}")
        print()


def simulate(table, tape_init, head_init, state_init, steps, x_max=40):
    """Simulate the TM and return tape history."""
    tape = list(tape_init)
    head = head_init
    state = state_init
    history = [(list(tape), head, state)]

    for t in range(steps):
        sym = tape[head]
        new_q, new_s, new_d = table[(state, sym)]
        tape[head] = new_s
        state = new_q
        head += new_d
        if head < 0 or head >= x_max:
            print(f"  Head out of bounds at step {t+1}: head={head}")
            break
        history.append((list(tape), head, state))

    return history


def print_tape_history(history, window=None):
    """Print tape evolution."""
    dir_sym = {0: '>', 1: '<'}
    for t, (tape, head, state) in enumerate(history):
        if window:
            lo, hi = window
        else:
            lo, hi = 0, len(tape)
        row = ""
        for x in range(lo, hi):
            sym_ch = str(tape[x]) if tape[x] not in (0, 7) else ('_' if tape[x] == 0 else '#')
            if x == head:
                row += f"[{sym_ch}]"
            else:
                row += f" {sym_ch} "
        print(f"  t={t:3d} {dir_sym[state]} {row}")


def iterative_deepening_search(t_start=10, t_end=80, t_step=5,
                               payload_in=None, payload_out=None,
                               target_state=None, target_head_region=None,
                               x_max=40, head_start=16):
    """
    Iterative deepening search: try increasing T values.
    """
    if payload_in is None:
        payload_in = [1, 1, 1, 4, 4, 2, 2, 3, 3]

    print("=" * 60)
    print("(2,8) Holographic TM — Z3 Iterative Deepening Search")
    print("=" * 60)
    print(f"Payload in:  {payload_in}")
    if payload_out:
        print(f"Payload out: {payload_out}")
    print(f"Target state: {target_state}")
    print(f"Target head region: {target_head_region}")
    print(f"Tape width: {x_max}")
    print(f"Search range: T={t_start}..{t_end} step {t_step}")
    print()

    for t_max in range(t_start, t_end + 1, t_step):
        print(f"--- Trying T_max = {t_max} ---")
        t0 = time.time()

        solver, info = build_solver_optimized(
            t_max=t_max,
            x_max=x_max,
            payload_in=payload_in,
            payload_out=payload_out,
            head_start=head_start,
            target_state=target_state,
            target_head_region=target_head_region,
        )

        result = solver.check()
        elapsed = time.time() - t0
        print(f"    Result: {result}  ({elapsed:.1f}s)")

        if result == sat:
            model = solver.model()
            table = extract_transition_table(model, info['Q_out'], info['S_out'], info['D_out'])

            print("\n*** SOLUTION FOUND ***")
            print_table(table)

            # Verify by simulation
            tape_init = []
            for x in range(x_max):
                if x < 16:
                    tape_init.append(0)
                elif x < 16 + len(payload_in):
                    tape_init.append(payload_in[x - 16])
                elif x < 25:
                    tape_init.append(3)
                else:
                    tape_init.append(7)

            history = simulate(table, tape_init, head_start, 0, t_max, x_max)
            print(f"\nTape evolution (window 10..30):")
            print_tape_history(history, window=(10, 30))

            # Show final payload region
            final_tape = history[-1][0]
            payload_region = final_tape[10:30]
            print(f"\nFinal payload region [10:30]: {payload_region}")
            print(f"Final head: {history[-1][1]}, state: {history[-1][2]}")

            # Save solution
            solution = {}
            for (q, sym), (qo, so, do) in table.items():
                solution[f"{q},{sym}"] = {"new_state": qo, "new_symbol": so, "direction": do}
            with open("solution_holographic.json", "w") as f:
                json.dump(solution, f, indent=2)
            print("\nSaved to solution_holographic.json")

            return table, history

        elif result == unsat:
            print(f"    UNSAT — no solution exists at T={t_max}")
        else:
            print(f"    UNKNOWN (timeout or resource limit)")

    print("\nSearch exhausted without finding a solution.")
    return None, None


def unconstrained_search(t_max=20, x_max=30, head_start=10):
    """
    Simpler search: just find ANY transition table that exhibits
    non-trivial sweeping behavior (no payload target).
    Useful for exploring the structure space.
    """
    print("=" * 60)
    print("(2,8) Holographic TM — Unconstrained Sweep Search")
    print("=" * 60)

    payload_in = [1, 2, 3, 4, 5, 6, 1, 2, 3]

    solver, info = build_solver_optimized(
        t_max=t_max,
        x_max=x_max,
        payload_in=payload_in,
        head_start=head_start,
        target_state=None,
        target_head_region=None,
    )

    # Just require the head to have moved significantly
    solver.add(info['head'][t_max] != head_start)
    # Require at least one boundary interaction
    # (head must reach x<=5 or x>=x_max-5 at some point)
    boundary_visits = []
    for t in range(t_max + 1):
        boundary_visits.append(Or(info['head'][t] <= 2, info['head'][t] >= x_max - 3))
    solver.add(Or(*boundary_visits))

    print(f"T_max={t_max}, tape width={x_max}")
    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract_transition_table(model, info['Q_out'], info['S_out'], info['D_out'])
        print_table(table)

        tape_init = [0]*10 + payload_in + [7]*(x_max - 10 - len(payload_in))
        history = simulate(table, tape_init, head_start, 0, t_max, x_max)
        print(f"\nTape evolution:")
        print_tape_history(history, window=(0, x_max))

        return table
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="(2,8) Holographic TM Search")
    parser.add_argument("--mode", choices=["search", "sweep", "quick"],
                        default="quick", help="Search mode")
    parser.add_argument("--t-start", type=int, default=10)
    parser.add_argument("--t-end", type=int, default=80)
    parser.add_argument("--t-step", type=int, default=5)
    parser.add_argument("--x-max", type=int, default=40)
    args = parser.parse_args()

    if args.mode == "quick":
        # Quick test: small horizon, no payload target, just check SAT
        print("Quick feasibility check (T=10, no target)...")
        solver, info = build_solver_optimized(
            t_max=10, x_max=30,
            payload_in=[1, 2, 3, 4, 5, 6],
            head_start=10,
        )
        t0 = time.time()
        result = solver.check()
        elapsed = time.time() - t0
        print(f"Result: {result} ({elapsed:.1f}s)")
        if result == sat:
            model = solver.model()
            table = extract_transition_table(model, info['Q_out'], info['S_out'], info['D_out'])
            print_table(table)

            # Simulate longer to see behavior
            tape_init = [0]*10 + [1,2,3,4,5,6] + [7]*14
            history = simulate(table, tape_init, 10, 0, 50, 30)
            print("\nExtended simulation (50 steps):")
            print_tape_history(history)

    elif args.mode == "sweep":
        unconstrained_search(t_max=20, x_max=30, head_start=10)

    elif args.mode == "search":
        # Full Minsky cycle search
        # Register A=3, B=0 -> Register A=2, B=1
        # Encoding: A as count of 1s, B as count of 4s, separated by marker 6
        payload_in  = [1, 1, 1, 6, 4, 4, 2, 2, 3]  # A=3, B=0 (4,4 are padding)
        payload_out = [1, 1, 6, 4, 4, 4, 2, 2, 3]   # A=2, B=1

        iterative_deepening_search(
            t_start=args.t_start,
            t_end=args.t_end,
            t_step=args.t_step,
            payload_in=payload_in,
            payload_out=payload_out,
            target_state=0,       # back to sweep right
            target_head_region=(0, 16),  # head near left boundary
            x_max=args.x_max,
            head_start=16,
        )
