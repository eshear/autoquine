#!/usr/bin/env python3
"""
(2,8) Holographic TM — Chained Minsky Simulation

Instead of isolated BMC scenarios, this uses a SINGLE long unrolling
with checkpoint assertions at specific time steps.

This guarantees cycles chain correctly because the tape state at the
end of one cycle IS the start state of the next cycle.

Strategy: find the per-cycle step count first with a small problem,
then verify with a full chain.
"""

import json
import sys
import time
from z3 import (
    Solver, Int, If, And, Or, Not, Sum, sat, unsat
)


def make_vars(solver, max_sing=4):
    """Create transition table variables."""
    Q = [[Int(f"Q{q}{s}") for s in range(8)] for q in range(2)]
    S = [[Int(f"S{q}{s}") for s in range(8)] for q in range(2)]
    D = [[Int(f"D{q}{s}") for s in range(8)] for q in range(2)]

    for q in range(2):
        for s in range(8):
            solver.add(Or(Q[q][s] == 0, Q[q][s] == 1))
            solver.add(S[q][s] >= 0, S[q][s] <= 7)
            solver.add(Or(D[q][s] == -1, D[q][s] == 1))

    # Boundary physics
    solver.add(Q[0][7] == 1, D[0][7] == -1, S[0][7] >= 1, S[0][7] <= 6)
    solver.add(Q[1][0] == 0, D[1][0] == 1, S[1][0] >= 1, S[1][0] <= 6)
    solver.add(Q[0][0] == 0, D[0][0] == 1, S[0][0] == 0)
    solver.add(Q[1][7] == 1, D[1][7] == -1, S[1][7] == 7)

    # Payload purity
    for q in range(2):
        for s in range(1, 7):
            solver.add(S[q][s] >= 1, S[q][s] <= 6)

    # Singularity bounds
    if max_sing < 6:
        rc = [If(And(Q[0][s] == 0, D[0][s] == 1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(rc) >= 6 - max_sing)
        lc = [If(And(Q[1][s] == 1, D[1][s] == -1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(lc) >= 6 - max_sing)

    return Q, S, D


def unroll(solver, Q, S, D, t_max, x_max, tape_init, head_start):
    """Unroll TM for t_max steps. Return tape/head/state arrays."""
    T = [[Int(f"t{t}x{x}") for x in range(x_max)] for t in range(t_max + 1)]
    H = [Int(f"h{t}") for t in range(t_max + 1)]
    ST = [Int(f"st{t}") for t in range(t_max + 1)]

    for x in range(x_max):
        solver.add(T[0][x] == tape_init[x])
    solver.add(ST[0] == 0, H[0] == head_start)

    for t in range(t_max):
        h = H[t]; st = ST[t]
        cur = T[t][0]
        for x in range(x_max - 1, -1, -1):
            cur = If(h == x, T[t][x], cur)
        nq = Q[0][0]; ns = S[0][0]; nd = D[0][0]
        for q in range(2):
            for sym in range(8):
                cond = And(st == q, cur == sym)
                nq = If(cond, Q[q][sym], nq)
                ns = If(cond, S[q][sym], ns)
                nd = If(cond, D[q][sym], nd)
        solver.add(ST[t+1] == nq, H[t+1] == h + nd)
        solver.add(H[t+1] >= 0, H[t+1] < x_max)
        for x in range(x_max):
            solver.add(T[t+1][x] == If(h == x, ns, T[t][x]))

    return T, H, ST


def extract(model, Q, S, D):
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
            if 1 <= s <= 6:
                if q == 0 and not (qo == 0 and do == 1):
                    tag = " *"
                if q == 1 and not (qo == 1 and do == -1):
                    tag = " *"
            print(f"{sn[q]:>4} {s:>4} -> {so:>4} {dn[do]:>4} {sn[qo]:>4}{tag}")
        print()


def sim(table, tape, head, state, steps, xm):
    tape = list(tape)
    hist = [(list(tape), head, state)]
    for _ in range(steps):
        if head < 0 or head >= xm:
            break
        nq, ns, nd = table[(state, tape[head])]
        tape[head] = ns
        state = nq
        head += nd
        hist.append((list(tape), head, state))
    return hist


def show(hist, window=None):
    ds = {0: '>', 1: '<'}
    for t, (tape, head, state) in enumerate(hist):
        lo = window[0] if window else 0
        hi = window[1] if window else len(tape)
        row = ""
        for x in range(lo, hi):
            ch = '_' if tape[x] == 0 else ('#' if tape[x] == 7 else str(tape[x]))
            row += f"[{ch}]" if x == head else f" {ch} "
        print(f"  t={t:3d} {ds.get(state,'?')} {row}")


def search_chained(steps_per_cycle=10, num_cycles=4, max_sing=4,
                   x_max=22, left_pad=7, timeout_s=600):
    """
    Search with chained unrolling.

    The full program (Minsky move-A-to-B then move-B-to-A) for A=2:
      [1,1,3] → [1,3,4] → [3,4,4] → [5,4,4] → [1,5,4] → [1,1,5] → [1,1,3]
    That's 6 steps for a complete cycle. Each step takes ~steps_per_cycle TM steps.

    Total TM steps = 6 * steps_per_cycle.
    """
    program = [
        [1, 1, 3],       # A=2, B=0, instr 0
        [1, 3, 4],       # A=1, B=1, instr 0
        [3, 4, 4],       # A=0, B=2, instr 0
        [5, 4, 4],       # A=0, B=2, instr 1
        [1, 5, 4],       # A=1, B=1, instr 1
        [1, 1, 5],       # A=2, B=0, instr 1
        # [1, 1, 3],     # back to start (cycle closes)
    ]

    total_steps = len(program) * steps_per_cycle
    pay_len = len(program[0])

    print("=" * 60)
    print(f"(2,8) Holographic TM — Chained Search")
    print(f"Steps/cycle={steps_per_cycle}, cycles={len(program)}, "
          f"total_T={total_steps}")
    print(f"max_sing={max_sing}, tape={x_max}, pad={left_pad}")
    print(f"Program cycle:")
    descs = [
        "I0: dec A (A=2→1)", "I0: dec A (A=1→0)", "I0→I1: switch",
        "I1: dec B (B=2→1)", "I1: dec B (B=1→0)", "I1→I0: switch"
    ]
    for i, (p, d) in enumerate(zip(program, descs)):
        print(f"  {i}: {p} ({d})")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)

    Q, S, D = make_vars(solver, max_sing=max_sing)

    # Build initial tape
    tape_init = [0] * left_pad + program[0] + [7] * (x_max - left_pad - pay_len)

    # Single long unrolling
    T, H, ST = unroll(solver, Q, S, D, total_steps, x_max, tape_init, left_pad)

    # Add checkpoint constraints at each cycle boundary
    for i, payload in enumerate(program[1:], 1):
        t_check = i * steps_per_cycle
        print(f"  Checkpoint t={t_check}: payload={payload}, state=0, head={left_pad}")
        for j, sym in enumerate(payload):
            solver.add(T[t_check][left_pad + j] == sym)
        # Head must be at start of payload, sweeping right
        solver.add(ST[t_check] == 0)
        solver.add(H[t_check] == left_pad)

    # Final checkpoint: must return to original config
    t_final = total_steps
    print(f"  Checkpoint t={t_final}: payload={program[0]} (cycle closes), "
          f"state=0, head={left_pad}")
    for j, sym in enumerate(program[0]):
        solver.add(T[t_final][left_pad + j] == sym)
    solver.add(ST[t_final] == 0)
    solver.add(H[t_final] == left_pad)

    print(f"\nSolving...")
    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract(model, Q, S, D)
        print_table(table)

        # Verify with simulation
        tape_init_list = [0]*left_pad + program[0] + [7]*(x_max-left_pad-pay_len)
        hist = sim(table, tape_init_list, left_pad, 0, total_steps, x_max)

        print(f"\n--- Full simulation ({total_steps} steps) ---")
        show(hist, window=(left_pad-3, left_pad+pay_len+5))

        # Check checkpoints
        print(f"\n--- Checkpoint verification ---")
        all_ok = True
        for i in range(len(program)):
            t_ck = i * steps_per_cycle
            expected = program[i]
            if t_ck < len(hist):
                tape_at, head_at, state_at = hist[t_ck]
                actual = tape_at[left_pad:left_pad+pay_len]
                ok = (actual == expected and head_at == left_pad and state_at == 0)
                status = "OK" if ok else "FAIL"
                if not ok:
                    all_ok = False
                print(f"  t={t_ck:3d} [{status}] payload={actual} head={head_at} "
                      f"state={state_at} (exp {expected})")
        # Check cycle closure
        t_ck = total_steps
        if t_ck < len(hist):
            tape_at, head_at, state_at = hist[t_ck]
            actual = tape_at[left_pad:left_pad+pay_len]
            ok = (actual == program[0] and head_at == left_pad and state_at == 0)
            if not ok:
                all_ok = False
            print(f"  t={t_ck:3d} [{'OK' if ok else 'FAIL'}] payload={actual} "
                  f"head={head_at} state={state_at} (exp {program[0]}, cycle close)")

        if all_ok:
            print("\n*** PERFECT CYCLE — ALL CHECKPOINTS VERIFIED ***")

            # Extended simulation to show it repeats
            x_big = 40
            tape_big = [0]*15 + program[0] + [7]*(x_big-15-pay_len)
            hist_ext = sim(table, tape_big, 15, 0, total_steps * 4, x_big)
            print(f"\n--- Extended simulation ({total_steps*4} steps) ---")
            for t in range(0, min(len(hist_ext), total_steps*4+1), steps_per_cycle):
                tape_at, head_at, state_at = hist_ext[t]
                l = 0
                while l < x_big and tape_at[l] == 0: l += 1
                r = x_big - 1
                while r >= 0 and tape_at[r] == 7: r -= 1
                payload = tape_at[l:r+1]
                ds = '>' if state_at == 0 else '<'
                a_ct = sum(1 for s in payload if s == 1)
                b_ct = sum(1 for s in payload if s == 4)
                mk = [s for s in payload if s in (3, 5)]
                instr = f"I{0 if 3 in mk else 1}" if mk else "??"
                print(f"  t={t:4d} {ds} h={head_at:2d} A={a_ct} B={b_ct} "
                      f"{instr} {payload}")

        # Save
        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_minsky_chained.json", "w") as f:
            json.dump(sol, f, indent=2)
        print("\nSaved to solution_minsky_chained.json")
        return table

    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=10, help="Steps per cycle")
    p.add_argument("--spc-max", type=int, default=20, help="Max steps per cycle")
    p.add_argument("--max-sing", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    print("=" * 60)
    print("Iterating over steps-per-cycle...")
    print("=" * 60)

    for spc in range(args.spc, args.spc_max + 1, 2):
        table = search_chained(
            steps_per_cycle=spc,
            max_sing=args.max_sing,
            x_max=22,
            left_pad=7,
            timeout_s=args.timeout,
        )
        if table:
            print(f"\n*** SOLUTION FOUND at spc={spc} ***")
            break
        print()
