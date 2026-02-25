#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Chained SAT Search

Key difference from holographic_sat.py: ONE continuous simulation of
total=6*spc steps, with checkpoints at each spc boundary. Boundary cells
are NOT reset between phases — they propagate naturally.

This allows the TM to use boundary cells as scratch space, which the
per-scenario model forbids.
"""

import time
import json
import sys
from pysat.solvers import Solver as SATSolver


class VarPool:
    def __init__(self):
        self.next_var = 1

    def new(self):
        v = self.next_var
        self.next_var += 1
        return v


def encode_chained(payloads, spc, eff_len, el, er):
    """
    Encode chained simulation as SAT.

    payloads: list of 7 payload patterns [start, p1, p2, ..., p6, start]
    spc: steps per phase
    eff_len: total effective tape length
    el: left boundary cells count
    er: right boundary cells count
    """
    pool = VarPool()
    clauses = []
    n_phases = len(payloads) - 1  # 6 phases
    total_steps = n_phases * spc
    start = el

    # Fixed boundary rules
    fixed = {
        (0, 0): (0, 0, 1),   # SR+0: pass through
        (1, 7): (1, 7, -1),  # SL+7: pass through
    }

    # Rule variables (same as before)
    # For payload rules (q,s) where q∈{0,1}, s∈{1..6}
    # and bounce rules (0,7), (1,0)
    rule_q = {}  # (q,s) -> var
    rule_w = {}  # (q,s,w) -> var (w=0..5 for write symbol 1..6)
    rule_d = {}  # (q,s) -> var (True=R, False=L)

    search_rules = [(q, s) for q in range(2) for s in range(1, 7)]
    search_rules += [(0, 7), (1, 0)]

    for (q, s) in search_rules:
        rule_q[(q, s)] = pool.new()
        for w in range(6):
            rule_w[(q, s, w)] = pool.new()
        rule_d[(q, s)] = pool.new()

        # One-hot on write symbol
        w_vars = [rule_w[(q, s, w)] for w in range(6)]
        clauses.append(w_vars[:])
        for i in range(6):
            for j in range(i + 1, 6):
                clauses.append([-w_vars[i], -w_vars[j]])

    # Bounce constraints
    clauses.append([rule_q[(0, 7)]])    # SR+7: new_q=1
    clauses.append([-rule_d[(0, 7)]])   # SR+7: dir=L
    clauses.append([-rule_q[(1, 0)]])   # SL+0: new_q=0
    clauses.append([rule_d[(1, 0)]])    # SL+0: dir=R

    # Trace variables: ONE continuous simulation
    tape_v = {}   # (t, x, s) -> var
    head_v = {}   # (t, x) -> var
    state_v = {}  # t -> var

    for t in range(total_steps + 1):
        state_v[t] = pool.new()
        for x in range(eff_len):
            head_v[(t, x)] = pool.new()
            for s in range(8):
                tape_v[(t, x, s)] = pool.new()

            # One-hot for tape symbol
            sym_vars = [tape_v[(t, x, s)] for s in range(8)]
            clauses.append(sym_vars[:])
            for i in range(8):
                for j in range(i + 1, 8):
                    clauses.append([-sym_vars[i], -sym_vars[j]])

        # One-hot for head position
        h_vars = [head_v[(t, x)] for x in range(eff_len)]
        clauses.append(h_vars[:])
        for i in range(eff_len):
            for j in range(i + 1, eff_len):
                clauses.append([-h_vars[i], -h_vars[j]])

    # Initial conditions
    init_tape = [0] * el + list(payloads[0]) + [7] * er
    clauses.append([-state_v[0]])  # state=0
    for x in range(eff_len):
        if x == start:
            clauses.append([head_v[(0, x)]])
        else:
            clauses.append([-head_v[(0, x)]])
        for s in range(8):
            if s == init_tape[x]:
                clauses.append([tape_v[(0, x, s)]])
            else:
                clauses.append([-tape_v[(0, x, s)]])

    # Transition constraints for all steps
    for t in range(total_steps):
        for x in range(eff_len):
            for q in range(2):
                for s in range(8):
                    cond_lits = [head_v[(t, x)], tape_v[(t, x, s)]]
                    if q == 1:
                        cond_lits.append(state_v[t])
                    else:
                        cond_lits.append(-state_v[t])
                    neg_cond = [-l for l in cond_lits]

                    key = (q, s)
                    if key in fixed:
                        nq, ns, nd = fixed[key]
                        new_x = x + nd
                        if new_x < 0 or new_x >= eff_len:
                            clauses.append([-l for l in cond_lits])
                            continue
                        _add_fixed(clauses, t, x, nq, ns, new_x,
                                   neg_cond, tape_v, head_v, state_v, eff_len)
                    elif key in [(qq, ss) for (qq, ss) in search_rules]:
                        _add_variable(clauses, t, x, q, s,
                                      neg_cond, tape_v, head_v, state_v,
                                      rule_q, rule_w, rule_d, eff_len)

    # Checkpoint conditions: at each phase boundary
    for phase in range(n_phases):
        t_ck = (phase + 1) * spc
        target = payloads[phase + 1]

        # State must be 0 (SR)
        clauses.append([-state_v[t_ck]])

        # Head at start
        for x in range(eff_len):
            if x == start:
                clauses.append([head_v[(t_ck, x)]])
            else:
                clauses.append([-head_v[(t_ck, x)]])

        # Payload cells must match
        for j, sym in enumerate(target):
            for s in range(8):
                if s == sym:
                    clauses.append([tape_v[(t_ck, start + j, s)]])
                else:
                    clauses.append([-tape_v[(t_ck, start + j, s)]])

        # Boundary cells are NOT constrained — they propagate naturally!

    n_vars = pool.next_var - 1
    return clauses, n_vars, rule_q, rule_w, rule_d, search_rules


def _add_fixed(clauses, t, x, nq, ns, new_x,
               neg_cond, tape_v, head_v, state_v, eff_len):
    if nq == 1:
        clauses.append(neg_cond + [state_v[t + 1]])
    else:
        clauses.append(neg_cond + [-state_v[t + 1]])

    clauses.append(neg_cond + [head_v[(t + 1, new_x)]])

    for s2 in range(8):
        if s2 == ns:
            clauses.append(neg_cond + [tape_v[(t + 1, x, s2)]])
        else:
            clauses.append(neg_cond + [-tape_v[(t + 1, x, s2)]])

    for x2 in range(eff_len):
        if x2 != x:
            for s2 in range(8):
                clauses.append(neg_cond + [-tape_v[(t, x2, s2)],
                               tape_v[(t + 1, x2, s2)]])
                clauses.append(neg_cond + [tape_v[(t, x2, s2)],
                               -tape_v[(t + 1, x2, s2)]])


def _add_variable(clauses, t, x, q, s,
                  neg_cond, tape_v, head_v, state_v,
                  rule_q, rule_w, rule_d, eff_len):
    key = (q, s)

    # State
    clauses.append(neg_cond + [-rule_q[key], state_v[t + 1]])
    clauses.append(neg_cond + [rule_q[key], -state_v[t + 1]])

    # Head
    new_x_r = x + 1
    new_x_l = x - 1

    if new_x_r >= eff_len and new_x_l < 0:
        clauses.append([-l for l in [-ll for ll in neg_cond]])
        return
    elif new_x_r >= eff_len:
        clauses.append(neg_cond + [-rule_d[key]])
        clauses.append(neg_cond + [head_v[(t + 1, new_x_l)]])
    elif new_x_l < 0:
        clauses.append(neg_cond + [rule_d[key]])
        clauses.append(neg_cond + [head_v[(t + 1, new_x_r)]])
    else:
        clauses.append(neg_cond + [-rule_d[key], head_v[(t + 1, new_x_r)]])
        clauses.append(neg_cond + [rule_d[key], head_v[(t + 1, new_x_l)]])

    # Write
    for w in range(6):
        ws = w + 1
        for s2 in range(8):
            if s2 == ws:
                clauses.append(neg_cond + [-rule_w[(key[0], key[1], w)],
                               tape_v[(t + 1, x, s2)]])
            else:
                clauses.append(neg_cond + [-rule_w[(key[0], key[1], w)],
                               -tape_v[(t + 1, x, s2)]])

    # Other positions unchanged
    for x2 in range(eff_len):
        if x2 != x:
            for s2 in range(8):
                clauses.append(neg_cond + [-tape_v[(t, x2, s2)],
                               tape_v[(t + 1, x2, s2)]])
                clauses.append(neg_cond + [tape_v[(t, x2, s2)],
                               -tape_v[(t + 1, x2, s2)]])


def decode_solution(model, rule_q, rule_w, rule_d, search_rules):
    table = {
        (0, 0): (0, 0, 1),
        (1, 7): (1, 7, -1),
    }
    for (q, s) in search_rules:
        nq = 1 if model[rule_q[(q, s)] - 1] > 0 else 0
        nd = 1 if model[rule_d[(q, s)] - 1] > 0 else -1
        ws = None
        for w in range(6):
            if model[rule_w[(q, s, w)] - 1] > 0:
                ws = w + 1
                break
        if ws is None:
            ws = 1
        table[(q, s)] = (nq, ws, nd)
    return table


def print_table(table):
    dn = {-1: 'L', 1: 'R'}
    sn = {0: 'SR', 1: 'SL'}
    print(f"\n{'St':>4} {'Sym':>4} -> {'Wr':>4} {'Dir':>4} {'Nxt':>4}")
    print("-" * 30)
    for q in range(2):
        for s in range(8):
            if (q, s) in table:
                qo, so, do = table[(q, s)]
                tag = ""
                if 1 <= s <= 6:
                    if q == 0 and not (qo == 0 and do == 1): tag = " *"
                    if q == 1 and not (qo == 1 and do == -1): tag = " *"
                print(f"{sn[q]:>4} {s:>4} -> {so:>4} {dn[do]:>4} {sn[qo]:>4}{tag}")
            else:
                print(f"{sn[q]:>4} {s:>4} -> {'?':>4} {'?':>4} {'?':>4}")
        print()


def verify_full(table, payloads, spc):
    print("\n--- Full tape verification ---")
    xm = 30; lp = 10
    tape = [0]*lp + list(payloads[0]) + [7]*(xm - lp - len(payloads[0]))
    h, s = lp, 0
    all_ok = True

    for phase in range(len(payloads) - 1):
        target = payloads[phase + 1]
        found = False
        for t in range(1, spc * 3 + 1):
            if h < 0 or h >= xm:
                break
            key = (s, tape[h])
            if key not in table:
                break
            nq, ns, nd = table[key]
            tape[h] = ns; s = nq; h += nd
            if h == lp and s == 0:
                actual = tape[lp:lp + len(target)]
                if actual == list(target):
                    print(f"  [OK] Phase {phase}: {payloads[phase]} → {actual} at t={t}")
                    found = True
                    break
        if not found:
            actual = tape[lp:lp + len(target)]
            print(f"  [FAIL] Phase {phase}: → {actual} (expected {target})")
            all_ok = False

    if all_ok:
        print("\n*** FULL TAPE VERIFIED ***")
        for a_val in [1, 2, 3, 4, 5, 8]:
            xb = 80; lp2 = 25
            pay = [1]*a_val + [3]
            tb = [0]*lp2 + pay + [7]*(xb - lp2 - len(pay))
            max_s = spc * (2*a_val + 2) * 5
            hb, sb = lp2, 0
            cyc = False
            for t in range(1, max_s + 1):
                if hb < 0 or hb >= xb: break
                key = (sb, tb[hb])
                if key not in table: break
                nq, ns, nd = table[key]
                tb[hb] = ns; sb = nq; hb += nd
                if hb == lp2 and sb == 0 and t > spc:
                    p = tb[lp2:lp2 + a_val + 1]
                    if p == [1]*a_val + [3]:
                        l = 0
                        while l < xb and tb[l] == 0: l += 1
                        if l == lp2:
                            print(f"  A={a_val}: CYCLE at t={t}")
                            cyc = True; break
            if not cyc:
                l = 0
                while l < xb and tb[l] == 0: l += 1
                r = xb - 1
                while r >= 0 and tb[r] == 7: r -= 1
                print(f"  A={a_val}: NO CYCLE. pay={tb[l:r+1][:12]}")
    return all_ok


def search(spc=8, el=1, er=0, solver_name='cadical153', encoding='old'):
    if encoding == 'old':
        payloads = [
            [1, 1, 3], [1, 3, 4], [3, 4, 4],
            [5, 4, 4], [1, 5, 4], [1, 1, 5], [1, 1, 3],
        ]
    elif encoding == 'compact':
        payloads = [
            [1, 1, 3], [1, 3, 2], [3, 2, 2],
            [4, 2, 2], [1, 4, 2], [1, 1, 4], [1, 1, 3],
        ]
    elif encoding == 'sliding':
        payloads = [
            [1, 1, 3], [1, 3, 1], [3, 1, 1],
            [5, 1, 1], [1, 5, 1], [1, 1, 5], [1, 1, 3],
        ]
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    eff_len = el + 3 + er
    eff_desc = f"{el}L+3P+{er}R (eff_len={eff_len})"

    print(f"\n{'='*60}")
    print(f"Chained SAT {encoding}: spc={spc} total={6*spc} eff={eff_desc}")
    print(f"{'='*60}")
    for i in range(6):
        print(f"  {i}: {payloads[i]} → {payloads[i+1]}")

    print(f"\nEncoding to SAT...")
    t0 = time.time()
    clauses, n_vars, rule_q, rule_w, rule_d, search_rules = \
        encode_chained(payloads, spc, eff_len, el, er)
    elapsed_enc = time.time() - t0
    print(f"  Variables: {n_vars:,}")
    print(f"  Clauses: {len(clauses):,}")
    print(f"  Encoding time: {elapsed_enc:.1f}s")

    print(f"\nSolving with {solver_name}...")
    t1 = time.time()
    solver = SATSolver(name=solver_name, bootstrap_with=clauses)
    result = solver.solve()
    elapsed_solve = time.time() - t1
    print(f"  Result: {'SAT' if result else 'UNSAT'} ({elapsed_solve:.1f}s)")

    if result:
        model = solver.get_model()
        table = decode_solution(model, rule_q, rule_w, rule_d, search_rules)
        print_table(table)
        verify_full(table, payloads, spc)

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open(f"solution_chained_sat_{encoding}_spc{spc}.json", "w") as f:
            json.dump(sol, f, indent=2)
        solver.delete()
        return table

    solver.delete()
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=6)
    p.add_argument("--spc-max", type=int, default=12)
    p.add_argument("--el", type=int, default=1)
    p.add_argument("--er", type=int, default=0)
    p.add_argument("--encoding", default="old",
                   choices=["old", "compact", "sliding"])
    p.add_argument("--solver", default="cadical153")
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1):
        for el, er in [(args.el, args.er)]:
            table = search(spc=spc, el=el, er=er,
                           solver_name=args.solver, encoding=args.encoding)
            if table:
                sys.exit(0)
