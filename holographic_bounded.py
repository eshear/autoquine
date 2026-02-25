#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Bounded Head Search

KEY INSIGHT from analyzing the A=1 solution:
- The TM never bounces off boundaries during the computation.
- All direction changes are via singularities within the payload.
- The left boundary (0) serves as a read-only "mirror": SR reads 0, writes 0, goes R.
- The TM oscillates between the boundary and payload cells.

This means we can CONSTRAIN the head to stay within [left_pad-1, left_pad+pay_len-1].
This dramatically reduces the search space since:
1. Only ~4 cells matter (not 14+)
2. Boundary cells are trivially clean (never touched)
3. The ITE chains are tiny
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


def add_bounded_scenario(solver, Q, S, D, sid, pay_in, pay_out,
                          spc, left_pad):
    """
    Add BMC scenario with head bounded to [left_pad-1, left_pad+pay_len-1].
    The tape only has pay_len+1 cells: one boundary 0, then the payload.
    No right boundary is ever reached.
    """
    pay_len = len(pay_in)
    # Effective tape: cell 0 = boundary (always 0), cells 1..pay_len = payload
    # Map: effective_x = real_x - (left_pad - 1)
    # So effective position 0 = left_pad - 1 (boundary 0)
    # effective position 1 = left_pad (first payload cell)
    # effective position pay_len = left_pad + pay_len - 1 (last payload cell)
    eff_len = pay_len + 1  # boundary cell + payload cells

    T = [[Int(f"s{sid}t{t}x{x}") for x in range(eff_len)]
         for t in range(spc + 1)]
    H = [Int(f"s{sid}h{t}") for t in range(spc + 1)]
    ST = [Int(f"s{sid}st{t}") for t in range(spc + 1)]

    # Initial tape: [0, pay_in[0], pay_in[1], ..., pay_in[n-1]]
    solver.add(T[0][0] == 0)
    for i, sym in enumerate(pay_in):
        solver.add(T[0][i + 1] == sym)
    # Start at effective position 1 (= left_pad), state SR
    solver.add(ST[0] == 0, H[0] == 1)

    # Unroll
    for t in range(spc):
        h = H[t]; st = ST[t]
        # Read current cell
        cur = T[t][0]
        for x in range(eff_len - 1, -1, -1):
            cur = If(h == x, T[t][x], cur)
        # Apply transition
        nq = Q[0][0]; ns = S[0][0]; nd = D[0][0]
        for q in range(2):
            for sym in range(8):
                cond = And(st == q, cur == sym)
                nq = If(cond, Q[q][sym], nq)
                ns = If(cond, S[q][sym], ns)
                nd = If(cond, D[q][sym], nd)
        solver.add(ST[t+1] == nq, H[t+1] == h + nd)
        # Head stays within bounds
        solver.add(H[t+1] >= 0, H[t+1] < eff_len)
        # Write
        for x in range(eff_len):
            solver.add(T[t+1][x] == If(h == x, ns, T[t][x]))

    # Target
    for j, sym in enumerate(pay_out):
        solver.add(T[spc][j + 1] == sym)
    solver.add(ST[spc] == 0, H[spc] == 1)
    # Boundary cell must still be 0
    solver.add(T[spc][0] == 0)


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


def search(max_sing=6, spc=10, timeout_s=300, encoding="sliding"):
    if encoding == "sliding":
        steps = [
            ([1, 1, 3], [1, 3, 1], 'I0: slide left (A=2→1)'),
            ([1, 3, 1], [3, 1, 1], 'I0: slide left (A=1→0)'),
            ([3, 1, 1], [5, 1, 1], 'I0→I1: change marker'),
            ([5, 1, 1], [1, 5, 1], 'I1: slide right (B=2→1)'),
            ([1, 5, 1], [1, 1, 5], 'I1: slide right (B=1→0)'),
            ([1, 1, 5], [1, 1, 3], 'I1→I0: change marker'),
        ]
    else:
        steps = [
            ([1, 1, 3], [1, 3, 4], 'I0: dec A (A=2→1)'),
            ([1, 3, 4], [3, 4, 4], 'I0: dec A (A=1→0)'),
            ([3, 4, 4], [5, 4, 4], 'I0→I1: switch'),
            ([5, 4, 4], [1, 5, 4], 'I1: dec B (B=2→1)'),
            ([1, 5, 4], [1, 1, 5], 'I1: dec B (B=1→0)'),
            ([1, 1, 5], [1, 1, 3], 'I1→I0: switch'),
        ]

    print("=" * 60)
    print(f"Bounded head search ({encoding})")
    print(f"spc={spc}, max_sing={max_sing}, head range=[0,3]")
    print("=" * 60)

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    for sid, (pi, po, desc) in enumerate(steps):
        add_bounded_scenario(solver, Q, S, D, sid, pi, po, spc, left_pad=4)
        print(f"  Added: {desc}: {pi} -> {po}")

    print(f"\nSolving ({len(steps)} scenarios, effective tape = 4 cells)...")
    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract(model, Q, S, D)
        print_table(table)

        # Verify with full simulation
        print("--- Verification (full tape) ---")
        all_ok = True
        for pi, po, desc in steps:
            x_max = 20
            left_pad = 8
            tape_init = [0]*left_pad + list(pi) + [7]*(x_max - left_pad - len(pi))
            hist = sim(table, tape_init, left_pad, 0, spc, x_max)
            final_tape, final_h, final_s = hist[-1]
            actual = final_tape[left_pad:left_pad + len(po)]
            ok = (actual == list(po) and final_h == left_pad and final_s == 0)
            left_ok = all(final_tape[x] == 0 for x in range(left_pad))
            right_ok = all(final_tape[x] == 7
                          for x in range(left_pad + len(po), x_max))
            full_ok = ok and left_ok and right_ok
            if not full_ok: all_ok = False
            print(f"  [{'OK' if full_ok else 'FAIL'}] {desc}: {pi} -> {actual} "
                  f"h={final_h} s={final_s}")

        if all_ok:
            print("\n*** ALL SCENARIOS VERIFIED ***")

            # Generalization
            for a_val in [1, 2, 3, 4, 5, 8]:
                x_big = 80
                lp = 30
                if encoding == "sliding":
                    payload = [1]*a_val + [3]
                else:
                    payload = [1]*a_val + [3]
                tape = [0]*lp + list(payload) + [7]*(x_big - lp - len(payload))
                max_s = spc * (2*a_val + 2) * 4
                h, s = lp, 0
                for t in range(1, max_s + 1):
                    if h < 0 or h >= x_big: break
                    nq, ns, nd = table[(s, tape[h])]
                    tape[h] = ns; s = nq; h += nd
                    if h == lp and s == 0 and t > spc:
                        pay = tape[lp:lp + a_val + 1]
                        if pay == [1]*a_val + [3]:
                            l = 0
                            while l < x_big and tape[l] == 0: l += 1
                            if l == lp:
                                print(f"  A={a_val}: CYCLE at t={t}")
                                break
                else:
                    l = 0
                    while l < x_big and tape[l] == 0: l += 1
                    r = x_big - 1
                    while r >= 0 and tape[r] == 7: r -= 1
                    print(f"  A={a_val}: NO CYCLE. pay={tape[l:r+1][:10]}")

            sol = {}
            for (q, s), (qo, so, do) in table.items():
                sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
            with open(f"solution_bounded_{encoding}.json", "w") as f:
                json.dump(sol, f, indent=2)
            return table

    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=8)
    p.add_argument("--spc-max", type=int, default=20)
    p.add_argument("--max-sing", type=int, default=6)
    p.add_argument("--encoding", choices=["sliding", "old"], default="old")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1, 2):
        print(f"\n{'#'*60}")
        table = search(max_sing=args.max_sing, spc=spc,
                       timeout_s=args.timeout, encoding=args.encoding)
        if table:
            print(f"\n*** SOLUTION at spc={spc} ***")
            break
        print()
