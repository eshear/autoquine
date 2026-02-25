#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Aggressive search strategies

Try multiple parameter combinations to find a sliding marker solution.
Key insight: try max_sing=6 (no singularity constraint) and small tape.
"""

import json
import time
from z3 import Solver, Int, If, And, Or, Sum, sat, unsat


def make_vars(solver, max_sing=6):
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


def add_scenario(solver, Q, S, D, sid, pay_in, pay_out, spc, x_max, left_pad,
                 cleanup=True):
    pay_len = len(pay_in)
    T = [[Int(f"s{sid}t{t}x{x}") for x in range(x_max)]
         for t in range(spc + 1)]
    H = [Int(f"s{sid}h{t}") for t in range(spc + 1)]
    ST = [Int(f"s{sid}st{t}") for t in range(spc + 1)]

    for x in range(x_max):
        if x < left_pad:
            solver.add(T[0][x] == 0)
        elif x < left_pad + pay_len:
            solver.add(T[0][x] == pay_in[x - left_pad])
        else:
            solver.add(T[0][x] == 7)
    solver.add(ST[0] == 0, H[0] == left_pad)

    for t in range(spc):
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

    for j, sym in enumerate(pay_out):
        solver.add(T[spc][left_pad + j] == sym)
    solver.add(ST[spc] == 0, H[spc] == left_pad)
    if cleanup:
        for x in range(left_pad):
            solver.add(T[spc][x] == 0)
        for x in range(left_pad + pay_len, x_max):
            solver.add(T[spc][x] == 7)


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


def test_generalization(table, spc):
    """Check if solution generalizes."""
    print("\n--- Generalization test ---")
    for a_val in [1, 2, 3, 4, 5, 8]:
        x_big = 80
        lp = 25
        payload = [1]*a_val + [3]
        tape = [0]*lp + payload + [7]*(x_big - lp - len(payload))
        max_s = spc * (2*a_val + 2) * 6
        hist = sim(table, tape, lp, 0, max_s, x_big)

        cycle_found = False
        for t in range(1, min(len(hist), max_s + 1)):
            tp, hd, st = hist[t]
            if hd == lp and st == 0:
                l = 0
                while l < x_big and tp[l] == 0: l += 1
                r = x_big - 1
                while r >= 0 and tp[r] == 7: r -= 1
                pay = tp[l:r+1]
                if (all(tp[x] == 0 for x in range(lp)) and
                    all(tp[x] == 7 for x in range(lp + max(len(pay), 1), x_big))):
                    if pay == [1]*a_val + [3]:
                        print(f"  A={a_val}: FULL CYCLE at t={t}")
                        cycle_found = True
                        break

        if not cycle_found:
            print(f"  A={a_val}: NO CYCLE after {max_s} steps")


def run_config(max_sing, spc, x_max, left_pad, timeout_s, steps):
    """Run a single configuration."""
    print(f"\n{'='*60}")
    print(f"Config: max_sing={max_sing}, spc={spc}, "
          f"tape={x_max}, pad={left_pad}")
    print(f"{'='*60}")

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    for sid, (pi, po, desc) in enumerate(steps):
        add_scenario(solver, Q, S, D, sid, pi, po, spc, x_max, left_pad)
        print(f"  Added: {desc}: {pi} -> {po}")

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract(model, Q, S, D)
        print_table(table)

        # Verify
        all_ok = True
        for pi, po, desc in steps:
            tape_init = [0]*left_pad + pi + [7]*(x_max - left_pad - len(pi))
            hist = sim(table, tape_init, left_pad, 0, spc, x_max)
            final = hist[-1]
            actual = final[0][left_pad:left_pad + len(po)]
            ok = (actual == po and final[1] == left_pad and final[2] == 0)
            if not ok: all_ok = False
            print(f"  [{'OK' if ok else 'FAIL'}] {desc}: {pi} -> {actual}")

        if all_ok:
            print("*** VERIFIED ***")
            test_generalization(table, spc)
            sol = {}
            for (q, s), (qo, so, do) in table.items():
                sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
            with open("solution_sliding_aggressive.json", "w") as f:
                json.dump(sol, f, indent=2)
            return table

    return None


if __name__ == "__main__":
    # Sliding marker A=2 cycle
    steps = [
        ([1, 1, 3], [1, 3, 1], 'I0: slide left'),
        ([1, 3, 1], [3, 1, 1], 'I0: slide left'),
        ([3, 1, 1], [5, 1, 1], 'I0→I1: change'),
        ([5, 1, 1], [1, 5, 1], 'I1: slide right'),
        ([1, 5, 1], [1, 1, 5], 'I1: slide right'),
        ([1, 1, 5], [1, 1, 3], 'I1→I0: change'),
    ]

    # Try configs from most promising to least
    configs = [
        # (max_sing, spc, x_max, left_pad, timeout_s)
        # Smallest feasible: no singularity limit, tiny tape
        (6, 10, 10, 3, 300),
        (6, 12, 10, 3, 300),
        (6, 10, 12, 3, 300),
        (6, 12, 12, 3, 300),
        (6, 14, 10, 3, 300),
        (6, 14, 12, 3, 300),
        # With 5 singularities
        (5, 10, 10, 3, 300),
        (5, 12, 10, 3, 300),
        (5, 12, 12, 3, 300),
        (5, 14, 12, 3, 300),
        # With 4 singularities
        (4, 12, 10, 3, 300),
        (4, 14, 10, 3, 300),
        (4, 14, 12, 3, 300),
    ]

    for i, (ms, spc, xm, lp, to) in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"Trial {i+1}/{len(configs)}")
        table = run_config(ms, spc, xm, lp, to, steps)
        if table:
            print(f"\n*** FOUND SOLUTION ***")
            break
