#!/usr/bin/env python3
"""
Reversible (2, 8) Holographic TM — SAT Search with Morita Constraints

Searches for a 2-state, 8-symbol TM that:
1. Simulates Minsky machine operations (A=1 and A=2 cycles)
2. Is reversible (satisfies Morita's local reversibility condition)

Morita's condition: for each (new_state, direction) group, all transitions
in that group must write distinct symbols. This guarantees the global map
is injective (reversible).

For our encoding:
- Fixed rules write symbols 0 and 7 (outside the {1..6} range of variable rules)
- So Morita reduces to: among variable rules in each group, write symbols are distinct
- 14 variable rules × 6 write values × 4 groups → manageable constraint set
"""

import time
import sys
import json
from pysat.solvers import Solver as SATSolver


class VarPool:
    def __init__(self):
        self.next_var = 1

    def new(self):
        v = self.next_var
        self.next_var += 1
        return v


def add_morita_constraints(clauses, rule_q, rule_w, rule_d, search_rules):
    """Add Morita's reversibility constraints to the clause set.

    For each pair of variable rules and each (new_state, direction) group,
    forbid both rules from being in the same group AND writing the same symbol.

    Each constraint is a single CNF clause:
      ¬(r1_in_group ∧ r2_in_group ∧ r1_writes_w ∧ r2_writes_w)
    """
    count = 0
    n = len(search_rules)

    for i in range(n):
        r1 = search_rules[i]
        rq1 = rule_q[r1]
        rd1 = rule_d[r1]

        for j in range(i + 1, n):
            r2 = search_rules[j]
            rq2 = rule_q[r2]
            rd2 = rule_d[r2]

            for nq_val in range(2):
                for d_val in range(2):
                    # Literal for "r NOT in this group"
                    # in_group = (rule_q matches nq_val) ∧ (rule_d matches d_val)
                    # ¬in_group = (rule_q doesn't match nq_val) ∨ (rule_d doesn't match d_val)
                    nq_lit1 = rq1 if nq_val == 0 else -rq1
                    nq_lit2 = rq2 if nq_val == 0 else -rq2
                    d_lit1 = rd1 if d_val == 0 else -rd1
                    d_lit2 = rd2 if d_val == 0 else -rd2

                    for w in range(6):
                        clause = [
                            nq_lit1, d_lit1,
                            nq_lit2, d_lit2,
                            -rule_w[(r1[0], r1[1], w)],
                            -rule_w[(r2[0], r2[1], w)],
                        ]
                        clauses.append(clause)
                        count += 1

    return count


def encode_reversible(payloads_list, spc, el):
    """Encode a reversible (2,8) TM search.

    payloads_list: list of (payloads, er) pairs, each defining a chained simulation
    """
    pool = VarPool()
    clauses = []

    fixed = {
        (0, 0): (0, 0, 1),
        (1, 7): (1, 7, -1),
    }

    # Rule variables for all search rules
    rule_q = {}
    rule_w = {}
    rule_d = {}
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
    clauses.append([rule_q[(0, 7)]])    # SR+7 → new_q=1
    clauses.append([-rule_d[(0, 7)]])   # SR+7 → dir=L
    clauses.append([-rule_q[(1, 0)]])   # SL+0 → new_q=0
    clauses.append([rule_d[(1, 0)]])    # SL+0 → dir=R

    # --- MORITA REVERSIBILITY CONSTRAINTS ---
    n_morita = add_morita_constraints(clauses, rule_q, rule_w, rule_d, search_rules)

    # Encode each simulation
    for sim_id, (payloads, er) in enumerate(payloads_list):
        pay_len = len(payloads[0])
        eff_len = el + pay_len + er
        start = el
        n_phases = len(payloads) - 1
        total = n_phases * spc

        tape_v = {}
        head_v = {}
        state_v = {}

        init_tape = [0] * el + list(payloads[0]) + [7] * er

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

        # Transition constraints
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
                            if nq == 1:
                                clauses.append(neg + [state_v[t + 1]])
                            else:
                                clauses.append(neg + [-state_v[t + 1]])
                            clauses.append(neg + [head_v[(t + 1, nx)]])
                            for s2 in range(8):
                                if s2 == ns:
                                    clauses.append(neg + [tape_v[(t + 1, x, s2)]])
                                else:
                                    clauses.append(neg + [-tape_v[(t + 1, x, s2)]])
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t + 1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t + 1, x2, s2)]])
                        elif key in [(qq, ss) for (qq, ss) in search_rules]:
                            # State
                            clauses.append(neg + [-rule_q[key], state_v[t + 1]])
                            clauses.append(neg + [rule_q[key], -state_v[t + 1]])
                            # Head
                            nxr = x + 1
                            nxl = x - 1
                            if nxr >= eff_len and nxl < 0:
                                clauses.append([-l for l in cond])
                                continue
                            elif nxr >= eff_len:
                                clauses.append(neg + [-rule_d[key]])
                                clauses.append(neg + [head_v[(t + 1, nxl)]])
                            elif nxl < 0:
                                clauses.append(neg + [rule_d[key]])
                                clauses.append(neg + [head_v[(t + 1, nxr)]])
                            else:
                                clauses.append(neg + [-rule_d[key], head_v[(t + 1, nxr)]])
                                clauses.append(neg + [rule_d[key], head_v[(t + 1, nxl)]])
                            # Write
                            for w in range(6):
                                ws = w + 1
                                for s2 in range(8):
                                    if s2 == ws:
                                        clauses.append(neg + [-rule_w[(key[0], key[1], w)], tape_v[(t + 1, x, s2)]])
                                    else:
                                        clauses.append(neg + [-rule_w[(key[0], key[1], w)], -tape_v[(t + 1, x, s2)]])
                            # Frame
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t + 1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t + 1, x2, s2)]])

        # Checkpoint constraints (payload + boundaries)
        for phase in range(n_phases):
            t_ck = (phase + 1) * spc
            target = payloads[phase + 1]
            tgt_len = len(target)
            clauses.append([-state_v[t_ck]])
            for x in range(eff_len):
                if x == start:
                    clauses.append([head_v[(t_ck, x)]])
                else:
                    clauses.append([-head_v[(t_ck, x)]])
            for j, sym in enumerate(target):
                for s in range(8):
                    if s == sym:
                        clauses.append([tape_v[(t_ck, start + j, s)]])
                    else:
                        clauses.append([-tape_v[(t_ck, start + j, s)]])
            # Clean boundaries
            for x in range(el):
                clauses.append([tape_v[(t_ck, x, 0)]])
            for x in range(el + tgt_len, eff_len):
                clauses.append([tape_v[(t_ck, x, 7)]])

    return clauses, pool, rule_q, rule_w, rule_d, search_rules, n_morita


def decode_solution(model, rule_q, rule_w, rule_d, search_rules):
    table = {(0, 0): (0, 0, 1), (1, 7): (1, 7, -1)}
    for (q, s) in search_rules:
        nq = 1 if model[rule_q[(q, s)] - 1] > 0 else 0
        nd = 1 if model[rule_d[(q, s)] - 1] > 0 else -1
        ws = 1
        for w in range(6):
            if model[rule_w[(q, s, w)] - 1] > 0:
                ws = w + 1
                break
        table[(q, s)] = (nq, ws, nd)
    return table


def check_morita(table):
    """Verify Morita's reversibility condition on a transition table."""
    from collections import defaultdict
    groups = defaultdict(list)
    for (q, s), (nq, ns, nd) in table.items():
        groups[(nq, nd)].append((q, s, ns))

    for (nq, nd), entries in groups.items():
        written = [e[2] for e in entries]
        if len(written) != len(set(written)):
            dn = {-1: 'L', 1: 'R'}
            print(f"  MORITA VIOLATION in group (nq={nq}, d={dn[nd]}): writes={written}")
            return False
    return True


def print_table(table):
    dn = {-1: 'L', 1: 'R'}
    sn = {0: 'SR', 1: 'SL'}
    for q in range(2):
        for s in range(8):
            if (q, s) in table:
                qo, so, do = table[(q, s)]
                tag = ""
                if 1 <= s <= 6:
                    if q == 0 and not (qo == 0 and do == 1):
                        tag = " *"
                    if q == 1 and not (qo == 1 and do == -1):
                        tag = " *"
                print(f"  {sn[q]:>2}+{s} -> {sn[qo]:>2} {so} {dn[do]}{tag}")


def verify(table, payloads, label, max_steps=500):
    """Verify a transition sequence on a full tape."""
    xm = 40
    lp = 15
    tape = [0] * lp + list(payloads[0]) + [7] * (xm - lp - len(payloads[0]))
    h, s = lp, 0
    for phase in range(len(payloads) - 1):
        target = payloads[phase + 1]
        found = False
        for t in range(1, max_steps + 1):
            if h < 0 or h >= xm:
                break
            key = (s, tape[h])
            if key not in table:
                break
            nq, ns, nd = table[key]
            tape[h] = ns
            s = nq
            h += nd
            if h == lp and s == 0:
                actual = tape[lp:lp + len(target)]
                if actual == list(target):
                    lb = all(tape[x] == 0 for x in range(lp))
                    rb = all(tape[x] == 7 for x in range(lp + len(target), xm))
                    tag = "OK" if lb and rb else "DIRTY"
                    print(f"  {label} Phase {phase}: {payloads[phase]} -> {actual} [{tag}] t={t}")
                    found = True
                    break
        if not found:
            actual = tape[lp:lp + len(payloads[phase]) + 4]
            print(f"  {label} Phase {phase}: FAIL -> {actual}")
            return False
    return True


def search(spc, el, er, solver_name='cadical153'):
    """Run the reversible search at a given spc."""
    # A=1 cycle (4 phases)
    a1 = [[1, 3], [3, 4], [5, 4], [1, 5], [1, 3]]
    # A=2 cycle (6 phases)
    a2 = [[1, 1, 3], [1, 3, 4], [3, 4, 4], [5, 4, 4], [1, 5, 4], [1, 1, 5], [1, 1, 3]]

    payloads_list = [
        (a1, er),
        (a2, er),
    ]

    print(f"\n{'=' * 60}")
    print(f"REVERSIBLE (2,8) SAT: spc={spc} el={el} er={er}")
    eff1 = el + 2 + er
    eff2 = el + 3 + er
    print(f"  A=1: {el}L+2P+{er}R = {eff1} cells, {4 * spc} steps")
    print(f"  A=2: {el}L+3P+{er}R = {eff2} cells, {6 * spc} steps")
    print(f"  Morita reversibility: ENFORCED")
    print(f"{'=' * 60}")

    print(f"\nEncoding...")
    t0 = time.time()
    clauses, pool, rq, rw, rd, sr, n_morita = encode_reversible(
        payloads_list, spc, el)
    enc_time = time.time() - t0

    n_vars = pool.next_var - 1
    n_clauses = len(clauses)
    print(f"  Variables: {n_vars:,}")
    print(f"  Clauses: {n_clauses:,}")
    print(f"  Morita clauses: {n_morita:,}")
    print(f"  Encoding time: {enc_time:.1f}s")

    print(f"\nSolving with {solver_name}...")
    t1 = time.time()
    solver = SATSolver(name=solver_name, bootstrap_with=clauses)
    result = solver.solve()
    solve_time = time.time() - t1
    print(f"  Result: {'SAT' if result else 'UNSAT'} ({solve_time:.1f}s)")

    if result:
        model = solver.get_model()
        table = decode_solution(model, rq, rw, rd, sr)

        print("\nTransition table:")
        print_table(table)

        print("\nMorita check:", "PASS" if check_morita(table) else "FAIL")

        # Show group breakdown
        from collections import defaultdict
        groups = defaultdict(list)
        for (q, s), (nq, ns, nd) in table.items():
            dn = {-1: 'L', 1: 'R'}
            groups[(nq, nd)].append(f"({q},{s})->w{ns}")
        print("\nGroup breakdown:")
        for (nq, nd) in sorted(groups.keys()):
            dn = {-1: 'L', 1: 'R'}
            print(f"  (nq={nq}, d={dn[nd]}): {groups[(nq, nd)]}")

        print("\nVerification:")
        ok1 = verify(table, a1, "A=1")
        ok2 = verify(table, a2, "A=2")

        if ok1 and ok2:
            print("\n*** REVERSIBLE SOLUTION FOUND ***")

            # Test generalization
            print("\nGeneralization:")
            for a_val in [3, 4, 5, 8]:
                pay = [1] * a_val + [3]
                verify(table, [pay, pay], f"A={a_val} identity", max_steps=a_val * 500)

            # Save
            sol = {}
            for (q, s), (qo, so, do) in table.items():
                sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
            sol["_meta"] = {"reversible": True, "spc": spc, "el": el, "er": er}
            with open("solution_reversible.json", "w") as f:
                json.dump(sol, f, indent=2)
            print(f"\nSaved to solution_reversible.json")

        solver.delete()
        return table

    solver.delete()
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reversible (2,8) holographic TM search")
    p.add_argument("--spc", type=int, default=8, help="Min steps per cycle phase")
    p.add_argument("--spc-max", type=int, default=20, help="Max steps per cycle phase")
    p.add_argument("--el", type=int, default=1, help="Left boundary cells")
    p.add_argument("--er", type=int, default=1, help="Right boundary cells")
    p.add_argument("--solver", default="cadical153")
    args = p.parse_args()

    print("Reversible (2,8) Holographic TM Search")
    print("=" * 60)
    print(f"Morita's condition: for each (new_state, dir) group,")
    print(f"all transitions write distinct symbols.")
    print(f"14 variable rules, 4 groups, 6 write values")
    print(f"Constraint: C(14,2) × 4 × 6 = {14*13//2 * 4 * 6} clauses")

    for spc in range(args.spc, args.spc_max + 1):
        table = search(spc, args.el, args.er, args.solver)
        if table:
            sys.exit(0)

    print(f"\nNo reversible solution found up to spc={args.spc_max}")
