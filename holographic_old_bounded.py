#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Old Encoding + Bounded Head + Incremental

The OLD encoding uses different symbols for A (1) and B (4) register units.
This naturally resolves the SL+1 conflict because:
- I0 context: SL encounters 1 (A unit) near marker 3
- I1 context: SL encounters 4 (B unit) near marker 5
Different symbols → different SL rules → no conflict.
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


def add_scenario(solver, Q, S, D, sid, pay_in, pay_out, spc, eff_tape, start):
    eff_len = len(eff_tape)
    T = [[Int(f"s{sid}t{t}x{x}") for x in range(eff_len)]
         for t in range(spc + 1)]
    H = [Int(f"s{sid}h{t}") for t in range(spc + 1)]
    ST = [Int(f"s{sid}st{t}") for t in range(spc + 1)]

    for x in range(eff_len):
        solver.add(T[0][x] == eff_tape[x])
    solver.add(ST[0] == 0, H[0] == start)

    for t in range(spc):
        h = H[t]; st = ST[t]
        cur = T[t][0]
        for x in range(eff_len - 1, -1, -1):
            cur = If(h == x, T[t][x], cur)
        nq = Q[0][0]; ns = S[0][0]; nd = D[0][0]
        for q in range(2):
            for sym in range(8):
                cond = And(st == q, cur == sym)
                nq = If(cond, Q[q][sym], nq)
                ns = If(cond, S[q][sym], ns)
                nd = If(cond, D[q][sym], nd)
        solver.add(ST[t+1] == nq, H[t+1] == h + nd)
        solver.add(H[t+1] >= 0, H[t+1] < eff_len)
        for x in range(eff_len):
            solver.add(T[t+1][x] == If(h == x, ns, T[t][x]))

    pay_len = len(pay_out)
    for j, sym in enumerate(pay_out):
        solver.add(T[spc][start + j] == sym)
    solver.add(ST[spc] == 0, H[spc] == start)
    # Boundary cleanup
    for x in range(start):
        solver.add(T[spc][x] == eff_tape[x])
    for x in range(start + pay_len, eff_len):
        solver.add(T[spc][x] == eff_tape[x])


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


def search(max_sing=6, spc=8, el=1, er=1, timeout_s=600):
    """
    Old encoding A=2 cycle:
    [1,1,3] → [1,3,4] → [3,4,4] → [5,4,4] → [1,5,4] → [1,1,5] → [1,1,3]
    """
    scenarios = [
        ([1, 1, 3], [1, 3, 4], 'I0: dec A (A=2→1)'),
        ([1, 3, 4], [3, 4, 4], 'I0: dec A (A=1→0)'),
        ([3, 4, 4], [5, 4, 4], 'I0→I1: switch'),
        ([5, 4, 4], [1, 5, 4], 'I1: dec B (B=2→1)'),
        ([1, 5, 4], [1, 1, 5], 'I1: dec B (B=1→0)'),
        ([1, 1, 5], [1, 1, 3], 'I1→I0: switch'),
    ]

    eff_desc = f"{el}L+3P+{er}R"
    print(f"{'='*60}")
    print(f"Old encoding bounded: spc={spc} eff={eff_desc} max_sing={max_sing}")
    print(f"{'='*60}")

    solver = Solver()
    solver.set("timeout", timeout_s * 1000)
    Q, S, D = make_vars(solver, max_sing=max_sing)

    for sid, (pi, po, desc) in enumerate(scenarios):
        left_b = [0] * el
        right_b = [7] * er
        eff_tape = left_b + list(pi) + right_b
        start = el

        solver.push()
        add_scenario(solver, Q, S, D, sid, pi, po, spc, eff_tape, start)

        t0 = time.time()
        result = solver.check()
        elapsed = time.time() - t0
        print(f"  [{sid}] {desc}: {result} ({elapsed:.1f}s)")

        if result == sat:
            pass  # Keep
        elif result == unsat:
            solver.pop()
            print(f"    UNSAT!")
            return None
        else:
            solver.pop()
            print(f"    Timeout!")
            return None

    model = solver.model()
    table = extract(model, Q, S, D)
    print(f"\n*** ALL 6 SCENARIOS SAT ***")
    print_table(table)

    # Verify on full tape
    print("--- Full tape verification ---")
    all_ok = True
    for pi, po, desc in scenarios:
        xm = 20; lp = 8
        tape = [0]*lp + list(pi) + [7]*(xm-lp-len(pi))
        # Try up to 3*spc steps
        h, s = lp, 0
        found = False
        for t in range(1, spc * 3 + 1):
            if h < 0 or h >= xm: break
            nq, ns, nd = table[(s, tape[h])]
            tape[h] = ns; s = nq; h += nd
            if t % spc == 0 and h == lp and s == 0:
                actual = tape[lp:lp+len(po)]
                if (actual == list(po) and
                    all(tape[x] == 0 for x in range(lp)) and
                    all(tape[x] == 7 for x in range(lp+len(po), xm))):
                    print(f"  [OK] {desc} at t={t}")
                    found = True
                    break
        if not found:
            print(f"  [FAIL] {desc}")
            all_ok = False

    if all_ok:
        print("\n*** FULL TAPE VERIFIED ***")
        # Generalization
        for a_val in [1, 2, 3, 4, 5, 8]:
            xb = 80; lp = 30
            pay = [1]*a_val + [3]
            tb = [0]*lp + pay + [7]*(xb-lp-len(pay))
            max_s = spc * (2*a_val+2) * 5
            h, s = lp, 0
            cyc = False
            for t in range(1, max_s + 1):
                if h < 0 or h >= xb: break
                nq, ns, nd = table[(s, tb[h])]
                tb[h] = ns; s = nq; h += nd
                if h == lp and s == 0 and t > spc:
                    p = tb[lp:lp+a_val+1]
                    if p == [1]*a_val+[3]:
                        l = 0
                        while l < xb and tb[l] == 0: l += 1
                        if l == lp:
                            print(f"  A={a_val}: CYCLE at t={t}")
                            cyc = True; break
            if not cyc:
                l = 0
                while l < xb and tb[l] == 0: l += 1
                r = xb - 1
                while r >= 0 and tb[r] == 7: r -= 1
                print(f"  A={a_val}: NO CYCLE. pay={tb[l:r+1][:12]}")

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_old_bounded.json", "w") as f:
            json.dump(sol, f, indent=2)
        print(f"\nSaved to solution_old_bounded.json")
        return table
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=8)
    p.add_argument("--spc-max", type=int, default=20)
    p.add_argument("--max-sing", type=int, default=6)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1, 2):
        for el, er in [(1, 0), (1, 1), (2, 0), (2, 1), (1, 2), (2, 2)]:
            print(f"\n{'#'*60}")
            table = search(max_sing=args.max_sing, spc=spc,
                           el=el, er=er, timeout_s=args.timeout)
            if table:
                print(f"\n*** SOLUTION: spc={spc}, {el}L+3P+{er}R ***")
                import sys; sys.exit(0)
