#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Bounded Head, Extended + Incremental

Try with 5-6 effective cells and incremental constraint building.
"""

import json
import time
from z3 import Solver, Int, If, And, Or, Sum, Not, sat, unsat


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


def add_scenario(solver, Q, S, D, sid, pay_in, pay_out, spc, eff_tape,
                 start_pos):
    """
    Add BMC scenario with bounded effective tape.
    eff_tape: list of initial symbols for the effective tape
    start_pos: starting position in effective tape
    """
    eff_len = len(eff_tape)

    T = [[Int(f"s{sid}t{t}x{x}") for x in range(eff_len)]
         for t in range(spc + 1)]
    H = [Int(f"s{sid}h{t}") for t in range(spc + 1)]
    ST = [Int(f"s{sid}st{t}") for t in range(spc + 1)]

    for x in range(eff_len):
        solver.add(T[0][x] == eff_tape[x])
    solver.add(ST[0] == 0, H[0] == start_pos)

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

    # Target: payload must match at the same positions
    pay_start = eff_tape.index(pay_in[0])  # Find where payload starts
    # Actually, just use start_pos since that's where the payload begins
    for j, sym in enumerate(pay_out):
        solver.add(T[spc][start_pos + j] == sym)
    solver.add(ST[spc] == 0, H[spc] == start_pos)

    # Boundary cells must be preserved
    for x in range(start_pos):
        solver.add(T[spc][x] == eff_tape[x])
    for x in range(start_pos + len(pay_out), eff_len):
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


def search_incremental(spc=10, max_sing=6, timeout_s=300, encoding="old"):
    if encoding == "sliding":
        scenarios = [
            ([1, 1, 3], [1, 3, 1], 'I0: slide left (A=2→1)'),
            ([1, 3, 1], [3, 1, 1], 'I0: slide left (A=1→0)'),
            ([3, 1, 1], [5, 1, 1], 'I0→I1: change marker'),
            ([5, 1, 1], [1, 5, 1], 'I1: slide right (B=2→1)'),
            ([1, 5, 1], [1, 1, 5], 'I1: slide right (B=1→0)'),
            ([1, 1, 5], [1, 1, 3], 'I1→I0: change marker'),
        ]
    else:
        scenarios = [
            ([1, 1, 3], [1, 3, 4], 'I0: dec A (A=2→1)'),
            ([1, 3, 4], [3, 4, 4], 'I0: dec A (A=1→0)'),
            ([3, 4, 4], [5, 4, 4], 'I0→I1: switch'),
            ([5, 4, 4], [1, 5, 4], 'I1: dec B (B=2→1)'),
            ([1, 5, 4], [1, 1, 5], 'I1: dec B (B=1→0)'),
            ([1, 1, 5], [1, 1, 3], 'I1→I0: switch'),
        ]

    # Try different effective tape sizes
    for extra_left in [1, 2]:
        for extra_right in [0, 1, 2]:
            eff_description = f"{extra_left}L+3P+{extra_right}R"
            print(f"\n{'='*60}")
            print(f"Incremental {encoding}: spc={spc}, eff={eff_description}")
            print(f"{'='*60}")

            solver = Solver()
            solver.set("timeout", timeout_s * 1000)
            Q, S, D = make_vars(solver, max_sing=max_sing)

            all_sat = True
            for sid, (pi, po, desc) in enumerate(scenarios):
                # Build effective tape
                left_boundary = [0] * extra_left
                right_boundary = [7] * extra_right
                eff_tape = left_boundary + list(pi) + right_boundary
                start_pos = extra_left

                solver.push()
                add_scenario(solver, Q, S, D, sid, pi, po, spc,
                             eff_tape, start_pos)

                t0 = time.time()
                result = solver.check()
                elapsed = time.time() - t0
                print(f"  [{sid}] {desc}: {result} ({elapsed:.1f}s)")

                if result == sat:
                    pass  # Keep scenario
                elif result == unsat:
                    solver.pop()
                    print(f"    UNSAT — need more steps or cells")
                    all_sat = False
                    break
                else:
                    solver.pop()
                    print(f"    Timeout")
                    all_sat = False
                    break

            if all_sat:
                model = solver.model()
                table = extract(model, Q, S, D)
                print(f"\n*** ALL {len(scenarios)} SCENARIOS SAT ***")
                print_table(table)

                # Full verification
                print("--- Full tape verification ---")
                all_ok = True
                for pi, po, desc in scenarios:
                    x_max = 20
                    left_pad = 8
                    tape_init = [0]*left_pad + list(pi) + [7]*(x_max-left_pad-len(pi))
                    tape_f, h_f, s_f = sim(table, tape_init, left_pad, 0, spc*3, x_max)
                    # Check at multiples of spc
                    found = False
                    tape_t = list(tape_init)
                    h_t, s_t = left_pad, 0
                    for t_step in range(1, spc * 3 + 1):
                        if h_t < 0 or h_t >= x_max: break
                        nq, ns, nd = table[(s_t, tape_t[h_t])]
                        tape_t[h_t] = ns; s_t = nq; h_t += nd
                        if t_step % spc == 0:
                            actual = tape_t[left_pad:left_pad+len(po)]
                            if (actual == list(po) and h_t == left_pad and s_t == 0
                                and all(tape_t[x] == 0 for x in range(left_pad))
                                and all(tape_t[x] == 7 for x in range(left_pad+len(po), x_max))):
                                print(f"  [OK] {desc}: at t={t_step}")
                                found = True
                                break
                    if not found:
                        all_ok = False
                        print(f"  [FAIL] {desc}")

                if all_ok:
                    print("\n*** VERIFIED ON FULL TAPE ***")

                    # Generalization
                    for a_val in [1, 2, 3, 4, 5]:
                        xb = 60; lp = 20
                        pay = [1]*a_val + [3]
                        tb = [0]*lp + pay + [7]*(xb-lp-len(pay))
                        max_s = spc*(2*a_val+2)*4
                        h, s = lp, 0
                        cyc = False
                        for t in range(1, max_s+1):
                            if h<0 or h>=xb: break
                            nq,ns,nd = table[(s,tb[h])]
                            tb[h]=ns; s=nq; h+=nd
                            if h==lp and s==0 and t>spc:
                                p = tb[lp:lp+a_val+1]
                                if p == [1]*a_val+[3]:
                                    l=0
                                    while l<xb and tb[l]==0: l+=1
                                    if l==lp:
                                        print(f"  A={a_val}: CYCLE at t={t}")
                                        cyc=True; break
                        if not cyc:
                            l=0
                            while l<xb and tb[l]==0: l+=1
                            r=xb-1
                            while r>=0 and tb[r]==7: r-=1
                            print(f"  A={a_val}: NO CYCLE. pay={tb[l:r+1][:10]}")

                    sol = {}
                    for (q, s), (qo, so, do) in table.items():
                        sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
                    with open(f"solution_bounded2_{encoding}.json", "w") as f:
                        json.dump(sol, f, indent=2)
                    return table

    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=8)
    p.add_argument("--spc-max", type=int, default=16)
    p.add_argument("--max-sing", type=int, default=6)
    p.add_argument("--encoding", choices=["sliding", "old", "both"], default="both")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    for encoding in (["sliding", "old"] if args.encoding == "both"
                     else [args.encoding]):
        for spc in range(args.spc, args.spc_max + 1, 2):
            print(f"\n{'#'*60}")
            print(f"Trying {encoding} spc={spc}")
            table = search_incremental(spc=spc, max_sing=args.max_sing,
                                       timeout_s=args.timeout,
                                       encoding=encoding)
            if table:
                print(f"\n*** SOLUTION: {encoding} spc={spc} ***")
                import sys
                sys.exit(0)
