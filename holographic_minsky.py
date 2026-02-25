#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Multi-Instruction Minsky Machine Search v2

Key insight: with 2 states, the TM carries instruction-type information
via intermediate symbols on the tape. Different markers (3 vs 5) trigger
different write patterns that encode which instruction to execute.

Strategy: incremental constraint building
1. Start with just dec-A/inc-B scenarios
2. Add dec-B/inc-A scenarios
3. Add zero-test branching scenarios
4. Use push/pop for efficient incremental solving

Encoding:
  0, 7 = boundaries
  1 = register A unit
  4 = register B unit
  3 = instruction 0 separator (dec A, inc B)
  5 = instruction 1 separator (dec B, inc A)
  2, 6 = transit/working symbols
"""

import json
import sys
import time
from z3 import (
    Solver, Int, If, And, Or, Not, Sum, Implies,
    sat, unsat, unknown
)


def make_transition_vars(solver, max_singularities=4):
    """Create transition table with flexible singularity constraints."""
    Q = [[Int(f"Qo_{q}_{s}") for s in range(8)] for q in range(2)]
    S = [[Int(f"So_{q}_{s}") for s in range(8)] for q in range(2)]
    D = [[Int(f"Do_{q}_{s}") for s in range(8)] for q in range(2)]

    for q in range(2):
        for s in range(8):
            solver.add(Or(Q[q][s] == 0, Q[q][s] == 1))
            solver.add(S[q][s] >= 0, S[q][s] <= 7)
            solver.add(Or(D[q][s] == -1, D[q][s] == 1))

    # Boundary physics
    solver.add(Q[0][7] == 1, D[0][7] == -1)
    solver.add(S[0][7] >= 1, S[0][7] <= 6)

    solver.add(Q[1][0] == 0, D[1][0] == 1)
    solver.add(S[1][0] >= 1, S[1][0] <= 6)

    solver.add(Q[0][0] == 0, D[0][0] == 1, S[0][0] == 0)
    solver.add(Q[1][7] == 1, D[1][7] == -1, S[1][7] == 7)

    # Payload purity
    for q in range(2):
        for s in range(1, 7):
            solver.add(S[q][s] >= 1, S[q][s] <= 6)

    # Singularity bound
    if max_singularities < 6:
        rc = [If(And(Q[0][s] == 0, D[0][s] == 1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(rc) >= 6 - max_singularities)
        lc = [If(And(Q[1][s] == 1, D[1][s] == -1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(lc) >= 6 - max_singularities)

    return Q, S, D


def add_scenario(solver, Q, S, D, sid, tape_init, head_start, t_max, x_max,
                 target_cells=None, target_state=None, target_head=None):
    """
    Add BMC scenario.
    target_cells: dict of {position: expected_symbol} at t=t_max
    target_head: required head position at t=t_max
    """
    tape = [[Int(f"s{sid}t{t}x{x}") for x in range(x_max)]
            for t in range(t_max + 1)]
    head = [Int(f"s{sid}h{t}") for t in range(t_max + 1)]
    state = [Int(f"s{sid}st{t}") for t in range(t_max + 1)]

    for x in range(x_max):
        solver.add(tape[0][x] == tape_init[x])
    solver.add(state[0] == 0, head[0] == head_start)

    for t in range(t_max):
        h = head[t]; st = state[t]
        cur = tape[t][0]
        for x in range(x_max - 1, -1, -1):
            cur = If(h == x, tape[t][x], cur)
        nq = Q[0][0]; ns = S[0][0]; nd = D[0][0]
        for q in range(2):
            for sym in range(8):
                cond = And(st == q, cur == sym)
                nq = If(cond, Q[q][sym], nq)
                ns = If(cond, S[q][sym], ns)
                nd = If(cond, D[q][sym], nd)
        solver.add(state[t+1] == nq, head[t+1] == h + nd)
        solver.add(head[t+1] >= 0, head[t+1] < x_max)
        for x in range(x_max):
            solver.add(tape[t+1][x] == If(h == x, ns, tape[t][x]))

    if target_cells:
        for pos, sym in target_cells.items():
            solver.add(tape[t_max][pos] == sym)
    if target_state is not None:
        solver.add(state[t_max] == target_state)
    if target_head is not None:
        solver.add(head[t_max] == target_head)

    return tape, head, state


def make_tape(lp, payload, xm):
    return [0]*lp + list(payload) + [7]*(xm - lp - len(payload))


def extract_table(model, Q, S, D):
    table = {}
    for q in range(2):
        for s in range(8):
            table[(q, s)] = (
                model.eval(Q[q][s]).as_long(),
                model.eval(S[q][s]).as_long(),
                model.eval(D[q][s]).as_long(),
            )
    return table


def print_table(table):
    dn = {-1: 'L', 1: 'R'}
    sn = {0: 'SR', 1: 'SL'}
    print(f"\n{'St':>4} {'Sym':>4} -> {'Wr':>4} {'Dir':>4} {'Nxt':>4}")
    print("-" * 30)
    for q in range(2):
        for s in range(8):
            qo, so, do = table[(q, s)]
            tag = ""
            if q == 0 and 1 <= s <= 6:
                tag = "" if (qo == 0 and do == 1) else " *"
            if q == 1 and 1 <= s <= 6:
                tag = "" if (qo == 1 and do == -1) else " *"
            print(f"{sn[q]:>4} {s:>4} -> {so:>4} {dn[do]:>4} {sn[qo]:>4}{tag}")
        print()


def simulate(table, tape_init, head, state, steps, x_max):
    tape = list(tape_init)
    history = [(list(tape), head, state)]
    for t in range(steps):
        if head < 0 or head >= x_max:
            break
        sym = tape[head]
        if (state, sym) not in table:
            break
        nq, ns, nd = table[(state, sym)]
        tape[head] = ns
        state = nq
        head += nd
        history.append((list(tape), head, state))
    return history


def print_history(history, window=None):
    ds = {0: '>', 1: '<'}
    for t, (tape, head, state) in enumerate(history):
        lo = window[0] if window else 0
        hi = window[1] if window else len(tape)
        row = ""
        for x in range(lo, hi):
            ch = str(tape[x])
            if tape[x] == 0: ch = '_'
            elif tape[x] == 7: ch = '#'
            row += f"[{ch}]" if x == head else f" {ch} "
        print(f"  t={t:3d} {ds.get(state,'?')} {row}")


def incremental_search(max_sing=4, x_max=20, left_pad=6, timeout_ms=300000):
    """
    Incremental search: add scenarios one by one using push/pop.
    """
    print("=" * 60)
    print("(2,8) Holographic TM — Incremental Minsky Search")
    print(f"max_sing={max_sing}, tape={x_max}, pad={left_pad}")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_ms)

    Q, S, D = make_transition_vars(solver, max_singularities=max_sing)

    # Phase 1: dec A, inc B (instruction 0) — 3 cycles
    phase1_scenarios = [
        ([1, 1, 1, 3],    {0: 1, 1: 1, 2: 3, 3: 4},     "I0: A=3,B=0 -> A=2,B=1"),
        ([1, 1, 3, 4],    {0: 1, 1: 3, 2: 4, 3: 4},     "I0: A=2,B=1 -> A=1,B=2"),
        ([1, 3, 4, 4],    {0: 3, 1: 4, 2: 4, 3: 4},     "I0: A=1,B=2 -> A=0,B=3"),
    ]

    # Phase 2: zero test + instruction switch
    phase2_scenarios = [
        ([3, 4, 4, 4],    {0: 5, 1: 4, 2: 4, 3: 4},     "I0->I1: A=0 switch"),
    ]

    # Phase 3: dec B, inc A (instruction 1) — 3 cycles
    phase3_scenarios = [
        ([5, 4, 4, 4],    {0: 1, 1: 5, 2: 4, 3: 4},     "I1: B=3,A=0 -> B=2,A=1"),
        ([1, 5, 4, 4],    {0: 1, 1: 1, 2: 5, 3: 4},     "I1: B=2,A=1 -> B=1,A=2"),
        ([1, 1, 5, 4],    {0: 1, 1: 1, 2: 1, 3: 5},     "I1: B=1,A=2 -> B=0,A=3"),
    ]

    # Phase 4: second zero test
    phase4_scenarios = [
        ([1, 1, 1, 5],    {0: 1, 1: 1, 2: 1, 3: 3},     "I1->I0: B=0 switch"),
    ]

    all_phases = [
        ("Phase 1: dec A / inc B", phase1_scenarios),
        ("Phase 2: A=0 → switch to I1", phase2_scenarios),
        ("Phase 3: dec B / inc A", phase3_scenarios),
        ("Phase 4: B=0 → switch to I0", phase4_scenarios),
    ]

    sid = 0
    for phase_name, scenarios in all_phases:
        print(f"\n--- {phase_name} ---")

        for pi, target_offsets, desc in scenarios:
            # Try increasing T
            found = False
            for t_max in range(8, 25, 2):
                solver.push()

                tape_init = make_tape(left_pad, pi, x_max)
                # Convert relative offsets to absolute positions
                target_cells = {left_pad + k: v for k, v in target_offsets.items()}

                add_scenario(solver, Q, S, D, sid, tape_init, left_pad,
                             t_max, x_max, target_cells=target_cells,
                             target_state=0)

                t0 = time.time()
                result = solver.check()
                elapsed = time.time() - t0
                print(f"  [{sid}] T={t_max:2d} {desc}: {result} ({elapsed:.1f}s)")

                if result == sat:
                    found = True
                    # Keep this scenario (don't pop)
                    sid += 1
                    break
                elif result == unsat:
                    solver.pop()
                    if t_max >= 22:
                        print(f"    UNSAT at all T — phase impossible!")
                        return None
                    continue
                else:
                    solver.pop()
                    print(f"    Timeout at T={t_max}, trying larger...")
                    continue

            if not found:
                print(f"  FAILED to find solution for: {desc}")
                return None

        # Show current table after each phase
        model = solver.model()
        table = extract_table(model, Q, S, D)
        print(f"\n  Current transition table after {phase_name}:")
        print_table(table)

    # All phases passed!
    print("\n" + "=" * 60)
    print("*** ALL PHASES SATISFIED ***")
    print("=" * 60)

    model = solver.model()
    table = extract_table(model, Q, S, D)
    print_table(table)

    # Verify all scenarios via simulation
    print("\n--- Verification ---")
    all_ok = True
    for phase_name, scenarios in all_phases:
        for pi, target_offsets, desc in scenarios:
            tape_init = make_tape(left_pad, pi, x_max)
            history = simulate(table, tape_init, left_pad, 0, 24, x_max)
            final = history[-1][0]
            actual = [final[left_pad + k] for k in sorted(target_offsets.keys())]
            expected = [target_offsets[k] for k in sorted(target_offsets.keys())]
            ok = actual == expected
            print(f"  [{'OK' if ok else 'FAIL'}] {desc}: {pi} -> {actual} (exp {expected})")
            if not ok:
                all_ok = False
                print_history(history, window=(left_pad-2, left_pad+8))

    if all_ok:
        # Full chained simulation
        print(f"\n--- Full chained simulation (A=3, B=0, instr 0, 500 steps) ---")
        x_big = 50
        tape_init = make_tape(15, [1, 1, 1, 3], x_big)
        history = simulate(table, tape_init, 15, 0, 500, x_big)

        # Print selected time steps
        checkpoints = list(range(0, min(len(history), 200), 1))
        for t in checkpoints:
            if t < len(history):
                tape, head, st = history[t]
                lo = max(0, min(head, 10) - 2)
                hi = min(x_big, max(head, 25) + 2)
                ds = {0: '>', 1: '<'}
                row = ""
                for x in range(lo, hi):
                    ch = str(tape[x])
                    if tape[x] == 0: ch = '_'
                    elif tape[x] == 7: ch = '#'
                    row += f"[{ch}]" if x == head else f" {ch} "
                print(f"  t={t:3d} {ds.get(st,'?')} {row}")

        # Analyze payload at checkpoints
        print("\n--- Payload analysis ---")
        for t in range(0, min(len(history), 500), 10):
            tape, head, st = history[t]
            left = 0
            while left < x_big and tape[left] == 0: left += 1
            right = x_big - 1
            while right >= 0 and tape[right] == 7: right -= 1
            payload = tape[left:right+1]
            a_count = payload.count(1)
            b_count = payload.count(4)
            markers = [(i, payload[i]) for i in range(len(payload)) if payload[i] in (3, 5)]
            instr = "I0" if any(s == 3 for _, s in markers) else ("I1" if any(s == 5 for _, s in markers) else "??")
            ds = '>' if st == 0 else '<'
            print(f"  t={t:3d} {ds} A={a_count} B={b_count} {instr} payload={payload}")

    # Save
    solution = {}
    for (q, s), (qo, so, do) in table.items():
        solution[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
    with open("solution_minsky.json", "w") as f:
        json.dump(solution, f, indent=2)
    print("\nSaved to solution_minsky.json")

    return table


def search_free_encoding(max_sing=4, x_max=20, left_pad=6,
                         t_per=15, timeout_ms=600000):
    """
    Alternative: don't fix the encoding. Let Z3 find its own payload format.
    Only require that a specific initial tape produces a specific final tape
    after enough steps, demonstrating a multi-step computation.

    We require: starting from tape A, after T1 steps → tape B,
    then from tape B, after T2 steps → tape C, etc.
    Where A→B→C→...→A forms a cycle.
    """
    print("=" * 60)
    print("(2,8) Holographic TM — Free Encoding Multi-Step Search")
    print(f"max_sing={max_sing}, t_per={t_per}, tape={x_max}")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_ms)

    Q, S, D = make_transition_vars(solver, max_singularities=max_sing)

    # Define "program states" via intermediate tape configs
    # State variables: what the payload looks like at each program step
    # We use variable payloads that the solver chooses
    num_prog_steps = 6  # full cycle
    pay_len = 4

    # Payload variables for each program step
    pay = [[Int(f"pay_{step}_{i}") for i in range(pay_len)]
           for step in range(num_prog_steps + 1)]

    # Domain for payload symbols
    for step in range(num_prog_steps + 1):
        for i in range(pay_len):
            solver.add(pay[step][i] >= 1, pay[step][i] <= 6)

    # The cycle must close: last payload == first payload
    for i in range(pay_len):
        solver.add(pay[num_prog_steps][i] == pay[0][i])

    # Diversity: consecutive payloads must differ
    for step in range(num_prog_steps):
        diffs = [If(pay[step][i] != pay[step+1][i], 1, 0) for i in range(pay_len)]
        solver.add(Sum(diffs) >= 1)

    # Additional diversity: the cycle should involve at least 3 distinct configs
    for step in range(num_prog_steps):
        for step2 in range(step + 2, min(step + num_prog_steps, num_prog_steps)):
            diffs = [If(pay[step][i] != pay[step2][i], 1, 0) for i in range(pay_len)]
            solver.add(Sum(diffs) >= 1)

    # Add BMC scenario for each program step
    for step in range(num_prog_steps):
        # Build initial tape from payload variables
        tape_vars = [[Int(f"fe{step}_t{t}x{x}") for x in range(x_max)]
                     for t in range(t_per + 1)]
        head_vars = [Int(f"fe{step}_h{t}") for t in range(t_per + 1)]
        state_vars = [Int(f"fe{step}_st{t}") for t in range(t_per + 1)]

        # Initial tape: 0s + payload + 7s
        for x in range(x_max):
            if x < left_pad:
                solver.add(tape_vars[0][x] == 0)
            elif x < left_pad + pay_len:
                solver.add(tape_vars[0][x] == pay[step][x - left_pad])
            else:
                solver.add(tape_vars[0][x] == 7)

        solver.add(state_vars[0] == 0, head_vars[0] == left_pad)

        # Unroll
        for t in range(t_per):
            h = head_vars[t]; st = state_vars[t]
            cur = tape_vars[t][0]
            for x in range(x_max - 1, -1, -1):
                cur = If(h == x, tape_vars[t][x], cur)
            nq = Q[0][0]; ns = S[0][0]; nd = D[0][0]
            for q in range(2):
                for sym in range(8):
                    cond = And(st == q, cur == sym)
                    nq = If(cond, Q[q][sym], nq)
                    ns = If(cond, S[q][sym], ns)
                    nd = If(cond, D[q][sym], nd)
            solver.add(state_vars[t+1] == nq, head_vars[t+1] == h + nd)
            solver.add(head_vars[t+1] >= 0, head_vars[t+1] < x_max)
            for x in range(x_max):
                solver.add(tape_vars[t+1][x] == If(h == x, ns, tape_vars[t][x]))

        # Target: next payload
        for i in range(pay_len):
            solver.add(tape_vars[t_per][left_pad + i] == pay[step + 1][i])
        solver.add(state_vars[t_per] == 0)

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"\nResult: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract_table(model, Q, S, D)

        # Extract discovered payloads
        print("\nDiscovered program cycle:")
        for step in range(num_prog_steps + 1):
            p = [model.eval(pay[step][i]).as_long() for i in range(pay_len)]
            print(f"  Step {step}: {p}")

        print_table(table)

        # Verify and simulate
        payloads = []
        for step in range(num_prog_steps):
            p = [model.eval(pay[step][i]).as_long() for i in range(pay_len)]
            payloads.append(p)

        print("\n--- Verification ---")
        for step in range(num_prog_steps):
            pi = payloads[step]
            po = payloads[(step + 1) % num_prog_steps]
            tape_init = make_tape(left_pad, pi, x_max)
            history = simulate(table, tape_init, left_pad, 0, t_per, x_max)
            final = history[-1][0]
            actual = final[left_pad:left_pad + pay_len]
            ok = actual == po
            print(f"  [{'OK' if ok else 'FAIL'}] Step {step}: {pi} -> {actual} (exp {po})")
            if not ok:
                print_history(history, window=(left_pad-2, left_pad+pay_len+4))

        # Extended chained
        print(f"\n--- Extended chained simulation (200 steps) ---")
        tape_init = make_tape(left_pad, payloads[0], x_max)
        history = simulate(table, tape_init, left_pad, 0, 200, x_max)
        for t in range(0, min(len(history), 200), 5):
            tape, head, st = history[t]
            ds = '>' if st == 0 else '<'
            lo = max(0, left_pad - 4)
            hi = min(x_max, left_pad + pay_len + 6)
            row = ""
            for x in range(lo, hi):
                ch = str(tape[x])
                if tape[x] == 0: ch = '_'
                elif tape[x] == 7: ch = '#'
                row += f"[{ch}]" if x == head else f" {ch} "
            print(f"  t={t:3d} {ds} {row}")

        solution = {}
        for (q, s), (qo, so, do) in table.items():
            solution[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_minsky.json", "w") as f:
            json.dump(solution, f, indent=2)

        return table
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["incremental", "free", "both"],
                        default="both")
    parser.add_argument("--max-sing", type=int, default=4)
    parser.add_argument("--t-per", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.mode in ("incremental", "both"):
        print("\n" + "#" * 60)
        print("# INCREMENTAL SEARCH")
        print("#" * 60)
        table = incremental_search(
            max_sing=args.max_sing,
            timeout_ms=args.timeout * 1000,
        )
        if table:
            print("\nINCREMENTAL SEARCH SUCCEEDED!")
            sys.exit(0)

    if args.mode in ("free", "both"):
        print("\n" + "#" * 60)
        print("# FREE ENCODING SEARCH")
        print("#" * 60)
        table = search_free_encoding(
            max_sing=args.max_sing,
            t_per=args.t_per,
            timeout_ms=args.timeout * 1000,
        )
        if table:
            print("\nFREE ENCODING SEARCH SUCCEEDED!")
