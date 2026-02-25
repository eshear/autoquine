#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Sliding Marker, Chained Unrolling

Single continuous TM execution with checkpoint assertions.
This ensures transitions chain correctly since the tape state
flows naturally from one step to the next.
"""

import json
import time
from z3 import Solver, Int, If, And, Or, Sum, sat, unsat


def make_vars(solver, max_sing=4):
    Q = [[Int(f"Q{q}{s}") for s in range(8)] for q in range(2)]
    S = [[Int(f"S{q}{s}") for s in range(8)] for q in range(2)]
    D = [[Int(f"D{q}{s}") for s in range(8)] for q in range(2)]
    for q in range(2):
        for s in range(8):
            solver.add(Or(Q[q][s] == 0, Q[q][s] == 1))
            solver.add(S[q][s] >= 0, S[q][s] <= 7)
            solver.add(Or(D[q][s] == -1, D[q][s] == 1))
    solver.add(Q[0][7] == 1, D[0][7] == -1, S[0][7] >= 1, S[0][7] <= 6)
    solver.add(Q[1][0] == 0, D[1][0] == 1, S[1][0] >= 1, S[1][0] <= 6)
    solver.add(Q[0][0] == 0, D[0][0] == 1, S[0][0] == 0)
    solver.add(Q[1][7] == 1, D[1][7] == -1, S[1][7] == 7)
    for q in range(2):
        for s in range(1, 7):
            solver.add(S[q][s] >= 1, S[q][s] <= 6)
    if max_sing < 6:
        rc = [If(And(Q[0][s] == 0, D[0][s] == 1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(rc) >= 6 - max_sing)
        lc = [If(And(Q[1][s] == 1, D[1][s] == -1), 1, 0) for s in range(1, 7)]
        solver.add(Sum(lc) >= 6 - max_sing)
    return Q, S, D


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
                if q == 0 and not (qo == 0 and do == 1): tag = " *"
                if q == 1 and not (qo == 1 and do == -1): tag = " *"
            print(f"{sn[q]:>4} {s:>4} -> {so:>4} {dn[do]:>4} {sn[qo]:>4}{tag}")
        print()


def sim(table, tape, head, state, steps, xm):
    tape = list(tape)
    hist = [(list(tape), head, state)]
    for _ in range(steps):
        if head < 0 or head >= xm:
            break
        nq, ns, nd = table[(state, tape[head])]
        tape[head] = ns; state = nq; head += nd
        hist.append((list(tape), head, state))
    return hist


def search_chained(spc=10, max_sing=4, x_max=16, left_pad=5, timeout_s=600):
    """Chained unrolling with sliding marker payloads."""
    program = [
        ([1, 1, 3], 'I0: slide left (A=2→1)'),
        ([1, 3, 1], 'I0: slide left (A=1→0)'),
        ([3, 1, 1], 'I0→I1: change marker'),
        ([5, 1, 1], 'I1: slide right (B=2→1)'),
        ([1, 5, 1], 'I1: slide right (B=1→0)'),
        ([1, 1, 5], 'I1→I0: change marker'),
    ]

    pay_len = 3
    total_steps = len(program) * spc

    print("=" * 60)
    print(f"Sliding Marker Chained: spc={spc}, total={total_steps}")
    print(f"max_sing={max_sing}, tape={x_max}, pad={left_pad}")
    for i, (p, d) in enumerate(program):
        print(f"  {i}: {p} ({d})")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    # Build initial tape
    tape_init = [0]*left_pad + program[0][0] + [7]*(x_max - left_pad - pay_len)

    # Single long unrolling
    T = [[Int(f"t{t}x{x}") for x in range(x_max)] for t in range(total_steps + 1)]
    H = [Int(f"h{t}") for t in range(total_steps + 1)]
    ST = [Int(f"st{t}") for t in range(total_steps + 1)]

    for x in range(x_max):
        solver.add(T[0][x] == tape_init[x])
    solver.add(ST[0] == 0, H[0] == left_pad)

    for t in range(total_steps):
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

    # Add checkpoint constraints + cleanup at each cycle boundary
    for i in range(1, len(program)):
        t_ck = i * spc
        payload = program[i][0]
        print(f"  Checkpoint t={t_ck}: payload={payload}")
        for j, sym in enumerate(payload):
            solver.add(T[t_ck][left_pad + j] == sym)
        solver.add(ST[t_ck] == 0, H[t_ck] == left_pad)
        # Full boundary cleanup
        for x in range(left_pad):
            solver.add(T[t_ck][x] == 0)
        for x in range(left_pad + pay_len, x_max):
            solver.add(T[t_ck][x] == 7)

    # Cycle closure: must return to [1, 1, 3]
    t_final = total_steps
    print(f"  Checkpoint t={t_final}: payload={program[0][0]} (close)")
    for j, sym in enumerate(program[0][0]):
        solver.add(T[t_final][left_pad + j] == sym)
    solver.add(ST[t_final] == 0, H[t_final] == left_pad)
    for x in range(left_pad):
        solver.add(T[t_final][x] == 0)
    for x in range(left_pad + pay_len, x_max):
        solver.add(T[t_final][x] == 7)

    print(f"\nSolving...")
    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract(model, Q, S, D)
        print_table(table)

        # Verify
        tape_init_list = list(tape_init)
        hist = sim(table, tape_init_list, left_pad, 0, total_steps, x_max)

        print("--- Checkpoint verification ---")
        all_ok = True
        all_payloads = [p for p, _ in program] + [program[0][0]]
        for i, payload in enumerate(all_payloads):
            t_ck = i * spc
            if t_ck < len(hist):
                tp, hd, st = hist[t_ck]
                actual = tp[left_pad:left_pad + pay_len]
                left_ok = all(tp[x] == 0 for x in range(left_pad))
                right_ok = all(tp[x] == 7 for x in range(left_pad + pay_len, x_max))
                ok = actual == payload and hd == left_pad and st == 0 and left_ok and right_ok
                if not ok: all_ok = False
                print(f"  t={t_ck:3d} [{'OK' if ok else 'FAIL'}] {actual} "
                      f"h={hd} s={st} bnd={'ok' if left_ok and right_ok else 'BAD'}")

        if all_ok:
            print("\n*** PERFECT CYCLE ***")

            # Generalization test
            for a_val in [1, 2, 3, 4, 5]:
                x_big = 60
                lp = 20
                payload = [1]*a_val + [3]
                tape_big = [0]*lp + payload + [7]*(x_big - lp - len(payload))
                max_s = spc * (2 * a_val + 2) * 3
                hist_g = sim(table, tape_big, lp, 0, max_s, x_big)

                print(f"\n--- A={a_val} generalization ({max_s} steps) ---")
                for t in range(0, min(len(hist_g), max_s + 1), spc):
                    tp, hd, st = hist_g[t]
                    l = 0
                    while l < x_big and tp[l] == 0: l += 1
                    r = x_big - 1
                    while r >= 0 and tp[r] == 7: r -= 1
                    pay = tp[l:r+1]
                    ds = '>' if st == 0 else '<'
                    clean = (hd == lp and st == 0 and
                             all(tp[x] == 0 for x in range(lp)) and
                             all(tp[x] == 7 for x in range(lp + len(pay), x_big)))
                    tag = ' <<' if clean else ''
                    print(f"  t={t:4d} {ds} h={hd:2d} pay={pay}{tag}")

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_sliding_chain.json", "w") as f:
            json.dump(sol, f, indent=2)
        print("\nSaved to solution_sliding_chain.json")
        return table

    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=8)
    p.add_argument("--spc-max", type=int, default=18)
    p.add_argument("--max-sing", type=int, default=4)
    p.add_argument("--x-max", type=int, default=16)
    p.add_argument("--pad", type=int, default=5)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1, 2):
        print(f"\n{'#'*60}")
        table = search_chained(spc=spc, max_sing=args.max_sing,
                               x_max=args.x_max, left_pad=args.pad,
                               timeout_s=args.timeout)
        if table:
            print(f"\n*** SOLUTION at spc={spc} ***")
            break
        print()
