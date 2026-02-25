#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Tiny tape chained search

Use the SMALLEST possible tape and chained unrolling.
Also try WITHOUT cleanup constraints first to see if the problem is SAT.
"""

import json
import time
from z3 import Solver, Int, If, And, Or, Sum, BitVec, sat, unsat


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
    for _ in range(steps):
        if head < 0 or head >= xm:
            break
        nq, ns, nd = table[(state, tape[head])]
        tape[head] = ns; state = nq; head += nd
    return tape, head, state


def search_chained(spc=10, max_sing=6, x_max=9, left_pad=2, timeout_s=300,
                   cleanup=True, encoding="sliding"):
    if encoding == "sliding":
        program = [
            [1, 1, 3], [1, 3, 1], [3, 1, 1],
            [5, 1, 1], [1, 5, 1], [1, 1, 5],
        ]
        descs = [
            "I0: slide L", "I0: slide L", "I0→I1",
            "I1: slide R", "I1: slide R", "I1→I0"
        ]
    else:
        program = [
            [1, 1, 3], [1, 3, 4], [3, 4, 4],
            [5, 4, 4], [1, 5, 4], [1, 1, 5],
        ]
        descs = [
            "I0: dec A", "I0: dec A", "I0→I1",
            "I1: dec B", "I1: dec B", "I1→I0"
        ]

    pay_len = 3
    total = len(program) * spc

    print(f"Chained {encoding}: spc={spc} total={total} "
          f"tape={x_max} pad={left_pad} cleanup={cleanup} max_sing={max_sing}")
    for i, (p, d) in enumerate(zip(program, descs)):
        print(f"  {i}: {p} ({d})")

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    tape_init = [0]*left_pad + program[0] + [7]*(x_max - left_pad - pay_len)
    T = [[Int(f"t{t}x{x}") for x in range(x_max)] for t in range(total + 1)]
    H = [Int(f"h{t}") for t in range(total + 1)]
    ST = [Int(f"st{t}") for t in range(total + 1)]

    for x in range(x_max):
        solver.add(T[0][x] == tape_init[x])
    solver.add(ST[0] == 0, H[0] == left_pad)

    for t in range(total):
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

    # Checkpoints
    for i in range(1, len(program)):
        t_ck = i * spc
        for j, sym in enumerate(program[i]):
            solver.add(T[t_ck][left_pad + j] == sym)
        solver.add(ST[t_ck] == 0, H[t_ck] == left_pad)
        if cleanup:
            for x in range(left_pad):
                solver.add(T[t_ck][x] == 0)
            for x in range(left_pad + pay_len, x_max):
                solver.add(T[t_ck][x] == 7)

    # Cycle close
    t_f = total
    for j, sym in enumerate(program[0]):
        solver.add(T[t_f][left_pad + j] == sym)
    solver.add(ST[t_f] == 0, H[t_f] == left_pad)
    if cleanup:
        for x in range(left_pad):
            solver.add(T[t_f][x] == 0)
        for x in range(left_pad + pay_len, x_max):
            solver.add(T[t_f][x] == 7)

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0
    print(f"Result: {result} ({elapsed:.1f}s)")

    if result == sat:
        model = solver.model()
        table = extract(model, Q, S, D)
        print_table(table)

        # Verify
        tape_v = list(tape_init)
        h_v, s_v = left_pad, 0
        all_ok = True
        for i in range(len(program)):
            t_ck = (i + 1) * spc
            expected = program[(i + 1) % len(program)]
            tape_v, h_v, s_v = sim(table, tape_v, h_v, s_v, spc, x_max)
            actual = tape_v[left_pad:left_pad + pay_len]
            ok = (actual == expected and h_v == left_pad and s_v == 0)
            if not ok: all_ok = False
            print(f"  t={t_ck:3d} [{'OK' if ok else 'FAIL'}] {actual}")

        if all_ok:
            print("*** VERIFIED ***")

            # Generalization
            for a_val in [1, 2, 3, 4, 5]:
                xb = 60; lp = 20
                if encoding == "sliding":
                    pay = [1]*a_val + [3]
                else:
                    pay = [1]*a_val + [3]
                tb = [0]*lp + pay + [7]*(xb - lp - len(pay))
                max_s = spc * (2*a_val + 2) * 4
                hb, sb = lp, 0
                cycle = False
                for t in range(1, max_s + 1):
                    if hb < 0 or hb >= xb: break
                    nq, ns, nd = table[(sb, tb[hb])]
                    tb[hb] = ns; sb = nq; hb += nd
                    if hb == lp and sb == 0 and t > spc:
                        p = tb[lp:lp + a_val + 1]
                        if p == [1]*a_val + [3]:
                            l = 0
                            while l < xb and tb[l] == 0: l += 1
                            if l == lp:
                                print(f"  A={a_val}: CYCLE at t={t}")
                                cycle = True
                                break
                if not cycle:
                    l = 0
                    while l < xb and tb[l] == 0: l += 1
                    r = xb - 1
                    while r >= 0 and tb[r] == 7: r -= 1
                    print(f"  A={a_val}: NO CYCLE. pay={tb[l:r+1][:10]}")

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open(f"solution_tiny_{encoding}.json", "w") as f:
            json.dump(sol, f, indent=2)
        return table
    return None


if __name__ == "__main__":
    configs = [
        # (encoding, spc, max_sing, x_max, pad, cleanup, timeout)
        # NO cleanup first (easier) — see if any solution exists
        ("sliding", 10, 6, 9, 2, False, 120),
        ("sliding", 12, 6, 9, 2, False, 120),
        ("old", 10, 6, 9, 2, False, 120),
        ("old", 12, 6, 9, 2, False, 120),
        # WITH cleanup
        ("sliding", 10, 6, 9, 2, True, 300),
        ("sliding", 12, 6, 9, 2, True, 300),
        ("sliding", 14, 6, 9, 2, True, 300),
        ("old", 10, 6, 9, 2, True, 300),
        ("old", 12, 6, 9, 2, True, 300),
        ("old", 14, 6, 9, 2, True, 300),
        # Slightly larger tape
        ("sliding", 10, 6, 11, 3, True, 300),
        ("sliding", 12, 6, 11, 3, True, 300),
        ("old", 10, 6, 11, 3, True, 300),
        ("old", 12, 6, 11, 3, True, 300),
    ]

    for i, (enc, spc, ms, xm, lp, cl, to) in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"Trial {i+1}/{len(configs)}")
        table = search_chained(spc=spc, max_sing=ms, x_max=xm,
                               left_pad=lp, timeout_s=to,
                               cleanup=cl, encoding=enc)
        if table:
            print(f"\n*** FOUND with {enc} spc={spc} ***")
            break
        print()
