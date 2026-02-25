#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Combined A=1 + A=2 SAT Search

Search for a transition table that handles BOTH:
1. The A=1 cycle on a 2-cell payload (4 phases, known solvable)
2. At least some A=2 transitions on a 3-cell payload

Uses chained SAT with payload-only checking at checkpoints.
"""

import time
import sys
from pysat.solvers import Solver as SATSolver


class VarPool:
    def __init__(self):
        self.next_var = 1
    def new(self):
        v = self.next_var
        self.next_var += 1
        return v


def encode_combined(a1_payloads, a2_payloads, spc, eff_len_1, eff_len_2, el):
    """
    Encode TWO chained simulations sharing the same rule variables.
    """
    pool = VarPool()
    clauses = []

    fixed = {
        (0, 0): (0, 0, 1),
        (1, 7): (1, 7, -1),
    }

    # Rule variables
    rule_q = {}; rule_w = {}; rule_d = {}
    search_rules = [(q, s) for q in range(2) for s in range(1, 7)]
    search_rules += [(0, 7), (1, 0)]

    for (q, s) in search_rules:
        rule_q[(q, s)] = pool.new()
        for w in range(6):
            rule_w[(q, s, w)] = pool.new()
        rule_d[(q, s)] = pool.new()

        w_vars = [rule_w[(q, s, w)] for w in range(6)]
        clauses.append(w_vars[:])
        for i in range(6):
            for j in range(i + 1, 6):
                clauses.append([-w_vars[i], -w_vars[j]])

    clauses.append([rule_q[(0, 7)]])
    clauses.append([-rule_d[(0, 7)]])
    clauses.append([-rule_q[(1, 0)]])
    clauses.append([rule_d[(1, 0)]])

    def encode_simulation(sim_id, payloads, eff_len, start, n_phases, spc_val):
        """Encode one chained simulation."""
        total = n_phases * spc_val
        tape_v = {}; head_v = {}; state_v = {}

        init_tape = [0] * el + list(payloads[0]) + [7] * (eff_len - el - len(payloads[0]))

        for t in range(total + 1):
            state_v[t] = pool.new()
            for x in range(eff_len):
                head_v[(t, x)] = pool.new()
                for s in range(8):
                    tape_v[(t, x, s)] = pool.new()
                sym_vars = [tape_v[(t, x, s)] for s in range(8)]
                clauses.append(sym_vars[:])
                for i in range(8):
                    for j in range(i + 1, 8):
                        clauses.append([-sym_vars[i], -sym_vars[j]])
            h_vars = [head_v[(t, x)] for x in range(eff_len)]
            clauses.append(h_vars[:])
            for i in range(eff_len):
                for j in range(i + 1, eff_len):
                    clauses.append([-h_vars[i], -h_vars[j]])

        # Initial conditions
        clauses.append([-state_v[0]])
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

        # Transitions
        for t in range(total):
            for x in range(eff_len):
                for q in range(2):
                    for s in range(8):
                        cond = [head_v[(t, x)], tape_v[(t, x, s)]]
                        if q == 1:
                            cond.append(state_v[t])
                        else:
                            cond.append(-state_v[t])
                        neg = [-l for l in cond]
                        key = (q, s)

                        if key in fixed:
                            nq, ns, nd = fixed[key]
                            nx = x + nd
                            if nx < 0 or nx >= eff_len:
                                clauses.append([-l for l in cond])
                                continue
                            if nq == 1: clauses.append(neg + [state_v[t+1]])
                            else: clauses.append(neg + [-state_v[t+1]])
                            clauses.append(neg + [head_v[(t+1, nx)]])
                            for s2 in range(8):
                                if s2 == ns: clauses.append(neg + [tape_v[(t+1, x, s2)]])
                                else: clauses.append(neg + [-tape_v[(t+1, x, s2)]])
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t+1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t+1, x2, s2)]])
                        elif key in [(qq,ss) for (qq,ss) in search_rules]:
                            # State
                            clauses.append(neg + [-rule_q[key], state_v[t+1]])
                            clauses.append(neg + [rule_q[key], -state_v[t+1]])
                            # Head
                            nxr = x + 1; nxl = x - 1
                            if nxr >= eff_len and nxl < 0:
                                clauses.append([-l for l in cond])
                                continue
                            elif nxr >= eff_len:
                                clauses.append(neg + [-rule_d[key]])
                                clauses.append(neg + [head_v[(t+1, nxl)]])
                            elif nxl < 0:
                                clauses.append(neg + [rule_d[key]])
                                clauses.append(neg + [head_v[(t+1, nxr)]])
                            else:
                                clauses.append(neg + [-rule_d[key], head_v[(t+1, nxr)]])
                                clauses.append(neg + [rule_d[key], head_v[(t+1, nxl)]])
                            # Write
                            for w in range(6):
                                ws = w + 1
                                for s2 in range(8):
                                    if s2 == ws: clauses.append(neg + [-rule_w[(key[0],key[1],w)], tape_v[(t+1, x, s2)]])
                                    else: clauses.append(neg + [-rule_w[(key[0],key[1],w)], -tape_v[(t+1, x, s2)]])
                            # Frame
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t+1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t+1, x2, s2)]])

        # Checkpoints (payload only, not boundaries)
        for phase in range(n_phases):
            t_ck = (phase + 1) * spc_val
            target = payloads[phase + 1]
            pay_len = len(target)
            clauses.append([-state_v[t_ck]])
            for x in range(eff_len):
                if x == start: clauses.append([head_v[(t_ck, x)]])
                else: clauses.append([-head_v[(t_ck, x)]])
            for j, sym in enumerate(target):
                for s in range(8):
                    if s == sym: clauses.append([tape_v[(t_ck, start + j, s)]])
                    else: clauses.append([-tape_v[(t_ck, start + j, s)]])
            # Also check boundaries are clean
            for x in range(el):
                clauses.append([tape_v[(t_ck, x, 0)]])  # left boundary = 0
            er = eff_len - el - pay_len
            for x in range(el + pay_len, eff_len):
                clauses.append([tape_v[(t_ck, x, 7)]])  # right boundary = 7

        return tape_v, head_v, state_v

    # A=1 simulation
    n_a1 = len(a1_payloads) - 1
    start_1 = el
    encode_simulation(0, a1_payloads, eff_len_1, start_1, n_a1, spc)

    # A=2 simulation
    n_a2 = len(a2_payloads) - 1
    start_2 = el
    encode_simulation(1, a2_payloads, eff_len_2, start_2, n_a2, spc)

    return clauses, pool, rule_q, rule_w, rule_d, search_rules


def decode_solution(model, rule_q, rule_w, rule_d, search_rules):
    table = {(0, 0): (0, 0, 1), (1, 7): (1, 7, -1)}
    for (q, s) in search_rules:
        nq = 1 if model[rule_q[(q, s)] - 1] > 0 else 0
        nd = 1 if model[rule_d[(q, s)] - 1] > 0 else -1
        ws = 1
        for w in range(6):
            if model[rule_w[(q, s, w)] - 1] > 0:
                ws = w + 1; break
        table[(q, s)] = (nq, ws, nd)
    return table


def print_table(table):
    dn = {-1: 'L', 1: 'R'}
    sn = {0: 'SR', 1: 'SL'}
    for q in range(2):
        for s in range(8):
            if (q, s) in table:
                qo, so, do = table[(q, s)]
                tag = " *" if 1 <= s <= 6 and not (q == 0 and qo == 0 and do == 1) and not (q == 1 and qo == 1 and do == -1) else ""
                print(f"  {sn[q]:>2}+{s} -> {sn[qo]:>2} {so} {dn[do]}{tag}")


def verify(table, payloads, label, max_steps=500):
    xm = 40; lp = 15
    tape = [0]*lp + list(payloads[0]) + [7]*(xm - lp - len(payloads[0]))
    h, s = lp, 0
    for phase in range(len(payloads) - 1):
        target = payloads[phase + 1]
        found = False
        for t in range(1, max_steps + 1):
            if h < 0 or h >= xm: break
            key = (s, tape[h])
            if key not in table: break
            nq, ns, nd = table[key]
            tape[h] = ns; s = nq; h += nd
            if h == lp and s == 0:
                actual = tape[lp:lp + len(target)]
                if actual == list(target):
                    # Check boundaries clean
                    lb = all(tape[x] == 0 for x in range(lp))
                    rb = all(tape[x] == 7 for x in range(lp + len(target), xm))
                    tag = "OK" if lb and rb else "DIRTY"
                    print(f"  {label} Phase {phase}: {payloads[phase]} → {actual} [{tag}] t={t}")
                    found = True; break
        if not found:
            actual = tape[lp:lp + len(payloads[phase]) + 4]
            print(f"  {label} Phase {phase}: FAIL → {actual}")
            return False
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=8)
    p.add_argument("--spc-max", type=int, default=14)
    p.add_argument("--el", type=int, default=1)
    args = p.parse_args()

    # A=1 old encoding cycle (4 phases)
    a1 = [[1,3], [3,4], [5,4], [1,5], [1,3]]
    # A=2 old encoding cycle (6 phases)
    a2 = [[1,1,3], [1,3,4], [3,4,4], [5,4,4], [1,5,4], [1,1,5], [1,1,3]]

    for spc in range(args.spc, args.spc_max + 1):
        el = args.el
        er_1 = 1  # right boundary cells for A=1
        er_2 = 1  # right boundary cells for A=2
        eff_len_1 = el + 2 + er_1
        eff_len_2 = el + 3 + er_2

        print(f"\n{'='*60}")
        print(f"Combined A=1+A=2 SAT: spc={spc}")
        print(f"  A=1: {el}L+2P+{er_1}R = {eff_len_1} cells")
        print(f"  A=2: {el}L+3P+{er_2}R = {eff_len_2} cells")
        print(f"  Clean boundaries enforced at checkpoints")
        print(f"{'='*60}")

        t0 = time.time()
        clauses, pool, rq, rw, rd, sr = encode_combined(
            a1, a2, spc, eff_len_1, eff_len_2, el)
        enc_time = time.time() - t0
        print(f"  Variables: {pool.next_var - 1:,}")
        print(f"  Clauses: {len(clauses):,}")
        print(f"  Encoding: {enc_time:.1f}s")

        t1 = time.time()
        solver = SATSolver(name='cadical153', bootstrap_with=clauses)
        result = solver.solve()
        solve_time = time.time() - t1
        print(f"  Result: {'SAT' if result else 'UNSAT'} ({solve_time:.1f}s)")

        if result:
            model = solver.get_model()
            table = decode_solution(model, rq, rw, rd, sr)
            print("\nTransition table:")
            print_table(table)
            print("\nVerification on full tape:")
            ok1 = verify(table, a1, "A=1")
            ok2 = verify(table, a2, "A=2")
            if ok1 and ok2:
                print("\n*** BOTH A=1 AND A=2 VERIFIED ***")
                # Test generalization
                for a_val in [3, 4, 5]:
                    pay = [1]*a_val + [3]
                    a_n = [pay]
                    # Build full cycle
                    for _ in range(2*a_val):
                        p = list(a_n[-1])
                        # Find marker
                        for i, x in enumerate(p):
                            if x in (3, 5):
                                if i > 0 and p[i-1] == 1:
                                    p[i-1] = 3 if x == 3 else 5
                                    p[i] = 4 if x == 3 else 1
                                    break
                        # Simplified - just verify cycle
                    verify(table, [pay, pay], f"A={a_val} identity", max_steps=5000)
            solver.delete()
            sys.exit(0)
        solver.delete()
