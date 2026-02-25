#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Split search: I0 first, then verify I1

Strategy: Find TM for I0 operations (3 scenarios), then check I1.
Due to potential symmetry, an I0 solution may also handle I1.
If not, incrementally add I1 constraints.

Also tries various spc values and mixed spc (different step counts per scenario).
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


def add_scenario(solver, Q, S, D, sid, pay_in, pay_out, spc, x_max, left_pad):
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
    # Full cleanup
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


def show_step(hist, t, left_pad, x_max):
    if t >= len(hist):
        return
    tp, hd, st = hist[t]
    l = 0
    while l < x_max and tp[l] == 0: l += 1
    r = x_max - 1
    while r >= 0 and tp[r] == 7: r -= 1
    pay = tp[l:r+1] if l <= r else []
    ds = '>' if st == 0 else '<'
    clean = (hd == left_pad and st == 0 and
             all(tp[x] == 0 for x in range(left_pad)) and
             all(tp[x] == 7 for x in range(left_pad + max(len(pay), 1), x_max)))
    tag = ' <<' if clean else ''
    print(f"  t={t:4d} {ds} h={hd:2d} pay={pay}{tag}")


def search_split(max_sing=4, spc=10, x_max=14, left_pad=4, timeout_s=300):
    """Find TM for I0, then verify/add I1."""
    # I0 scenarios (marker 3, slide left)
    i0_steps = [
        ([1, 1, 3], [1, 3, 1], 'I0: slide left (A=2→1)'),
        ([1, 3, 1], [3, 1, 1], 'I0: slide left (A=1→0)'),
        ([3, 1, 1], [5, 1, 1], 'I0→I1: change marker'),
    ]

    # I1 scenarios (marker 5, slide right)
    i1_steps = [
        ([5, 1, 1], [1, 5, 1], 'I1: slide right (B=2→1)'),
        ([1, 5, 1], [1, 1, 5], 'I1: slide right (B=1→0)'),
        ([1, 1, 5], [1, 1, 3], 'I1→I0: change marker'),
    ]

    print("=" * 60)
    print(f"Split search: spc={spc}, max_sing={max_sing}, "
          f"tape={x_max}, pad={left_pad}")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    # Phase 1: Add I0 scenarios
    print("\n--- Phase 1: I0 only ---")
    for sid, (pi, po, desc) in enumerate(i0_steps):
        add_scenario(solver, Q, S, D, sid, pi, po, spc, x_max, left_pad)
        print(f"  Added: {desc}")

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"I0-only result: {result} ({elapsed:.1f}s)")

    if result != sat:
        return None

    model = solver.model()
    table_i0 = extract(model, Q, S, D)
    print_table(table_i0)

    # Check if I0 solution also handles I1
    print("--- Checking I1 with I0 solution ---")
    i1_works = True
    for pi, po, desc in i1_steps:
        tape_init = [0]*left_pad + pi + [7]*(x_max - left_pad - len(pi))
        hist = sim(table_i0, tape_init, left_pad, 0, spc * 3, x_max)
        found = False
        for check_t in range(1, len(hist)):
            tp, hd, st = hist[check_t]
            actual = tp[left_pad:left_pad + len(po)]
            if actual == po and hd == left_pad and st == 0:
                left_ok = all(tp[x] == 0 for x in range(left_pad))
                right_ok = all(tp[x] == 7 for x in range(left_pad + len(po), x_max))
                if left_ok and right_ok:
                    print(f"  [OK] {desc}: found at t={check_t}")
                    found = True
                    break
        if not found:
            print(f"  [FAIL] {desc}: not found in {spc*3} steps")
            i1_works = False

    if i1_works:
        print("\n*** I0 solution ALSO handles I1! ***")
        _test_extended(table_i0, spc, left_pad)
        _save(table_i0, "solution_sliding_split.json")
        return table_i0

    # Phase 2: Add I1 scenarios incrementally
    print("\n--- Phase 2: Adding I1 scenarios ---")
    for sid2, (pi, po, desc) in enumerate(i1_steps):
        solver.push()
        add_scenario(solver, Q, S, D, len(i0_steps) + sid2,
                     pi, po, spc, x_max, left_pad)

        t0 = time.time()
        result = solver.check()
        elapsed = time.time() - t0
        print(f"  [{len(i0_steps) + sid2}] {desc}: {result} ({elapsed:.1f}s)")

        if result == sat:
            pass  # Keep scenario
        elif result == unsat:
            solver.pop()
            print(f"    UNSAT — I1 incompatible at spc={spc}")
            return None
        else:
            solver.pop()
            print(f"    Timeout")
            return None

    # All 6 scenarios satisfied!
    model = solver.model()
    table = extract(model, Q, S, D)
    print("\n*** ALL 6 SCENARIOS SATISFIED ***")
    print_table(table)

    # Verify
    print("--- Verification ---")
    all_ok = True
    for pi, po, desc in i0_steps + i1_steps:
        tape_init = [0]*left_pad + pi + [7]*(x_max - left_pad - len(pi))
        hist = sim(table, tape_init, left_pad, 0, spc, x_max)
        final = hist[-1]
        actual = final[0][left_pad:left_pad + len(po)]
        ok = actual == po and final[1] == left_pad and final[2] == 0
        left_ok = all(final[0][x] == 0 for x in range(left_pad))
        right_ok = all(final[0][x] == 7 for x in range(left_pad + len(po), x_max))
        full_ok = ok and left_ok and right_ok
        if not full_ok: all_ok = False
        print(f"  [{'OK' if full_ok else 'FAIL'}] {desc}: {pi} -> {actual}")

    if all_ok:
        _test_extended(table, spc, left_pad)
        _save(table, "solution_sliding_split.json")
        return table

    return None


def _test_extended(table, spc, left_pad):
    """Test generalization."""
    for a_val in [1, 2, 3, 4, 5, 8]:
        x_big = 80
        lp = 25
        payload = [1]*a_val + [3]
        tape_big = [0]*lp + payload + [7]*(x_big - lp - len(payload))
        max_s = spc * (2 * a_val + 2) * 5
        hist = sim(table, tape_big, lp, 0, max_s, x_big)

        print(f"\n--- A={a_val} ({max_s} steps) ---")
        cycle_found = False
        for t in range(1, min(len(hist), max_s + 1)):
            tp, hd, st = hist[t]
            if hd == lp and st == 0:
                l = 0
                while l < x_big and tp[l] == 0: l += 1
                r = x_big - 1
                while r >= 0 and tp[r] == 7: r -= 1
                pay = tp[l:r+1]
                clean = (all(tp[x] == 0 for x in range(lp)) and
                         all(tp[x] == 7 for x in range(lp + max(len(pay), 1), x_big)))
                if clean:
                    tag = ''
                    if pay == [1]*a_val + [3]:
                        tag = ' FULL CYCLE!'
                        cycle_found = True
                    print(f"  t={t:4d} > h={hd:2d} pay={pay}{tag}")
                    if cycle_found:
                        break

        if not cycle_found:
            tp, hd, st = hist[-1]
            l = 0
            while l < x_big and tp[l] == 0: l += 1
            r = x_big - 1
            while r >= 0 and tp[r] == 7: r -= 1
            pay = tp[l:r+1]
            print(f"  NO CYCLE after {max_s} steps. Last: pay={pay}")


def _save(table, filename):
    sol = {}
    for (q, s), (qo, so, do) in table.items():
        sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
    with open(filename, "w") as f:
        json.dump(sol, f, indent=2)
    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=10)
    p.add_argument("--spc-max", type=int, default=18)
    p.add_argument("--max-sing", type=int, default=4)
    p.add_argument("--x-max", type=int, default=14)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1, 2):
        print(f"\n{'#'*60}")
        table = search_split(max_sing=args.max_sing, spc=spc,
                              x_max=args.x_max, left_pad=args.pad,
                              timeout_s=args.timeout)
        if table:
            print(f"\n*** SOLUTION at spc={spc} ***")
            break
        print()
