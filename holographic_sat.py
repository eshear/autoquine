#!/usr/bin/env python3
"""
(2, 8) Holographic TM — SAT Solver Based Search

Encode the TM search as a propositional SAT problem using PySAT.
This should be MUCH faster than Z3 (SMT) or DFS for this finite-domain problem.

Variables:
- Rule variables: for each (q,s) pair, encode (new_q, write_s, dir) as booleans
- Trace variables: for each scenario, time step, encode tape/head/state as booleans
"""

import time
import json
import sys
import itertools
from pysat.solvers import Solver as SATSolver
from pysat.card import CardEnc, EncType


class VarPool:
    """Manage SAT variable allocation."""
    def __init__(self):
        self.next_var = 1
        self.names = {}

    def new(self, name=None):
        v = self.next_var
        self.next_var += 1
        if name:
            self.names[v] = name
        return v

    def new_array(self, n, prefix=""):
        return [self.new(f"{prefix}_{i}") for i in range(n)]


def encode_holographic_sat(scenarios, eff_tapes, starts, spc, eff_len):
    """
    Encode the holographic TM search as SAT.

    Returns (clauses, pool, rule_vars, trace_vars).
    """
    pool = VarPool()
    clauses = []

    # Number of payload symbol options: 1..6 = indices 0..5 (6 options)
    # For each payload rule (q, s) where q∈{0,1}, s∈{1..6}:
    #   new_q ∈ {0,1}  -> 1 bool
    #   write_s ∈ {1..6} -> 6 bools (one-hot)
    #   dir ∈ {-1,1} -> 1 bool (0=left, 1=right)

    # Rule variables
    # rule_q[q][s] = bool for new_q (True=1, False=0)
    # rule_w[q][s][w] = bool for write_symbol w (one-hot, w=0..5 meaning symbol 1..6)
    # rule_d[q][s] = bool for direction (True=right, False=left)
    rule_q = {}  # (q, s) -> var
    rule_w = {}  # (q, s, w) -> var  (w=0..5 for symbols 1..6)
    rule_d = {}  # (q, s) -> var

    # Fixed rules
    fixed_rules = {
        (0, 0): (0, 0, 1),    # SR+0: pass through
        (1, 7): (1, 7, -1),   # SL+7: pass through
    }

    # Payload rules to search: (q, s) for q∈{0,1}, s∈{1..6}
    free_rules = [(q, s) for q in range(2) for s in range(1, 7)]

    # Bounce rules: partially constrained
    # SR+7 → (1, W, -1): new_q=1, dir=L, write∈{1..6}
    # SL+0 → (0, V, 1): new_q=0, dir=R, write∈{1..6}
    bounce_rules = [(0, 7), (1, 0)]

    all_search_rules = free_rules + bounce_rules

    for (q, s) in all_search_rules:
        rule_q[(q, s)] = pool.new(f"rq_{q}_{s}")
        for w in range(6):
            rule_w[(q, s, w)] = pool.new(f"rw_{q}_{s}_{w}")
        rule_d[(q, s)] = pool.new(f"rd_{q}_{s}")

        # One-hot constraint on write symbol
        w_vars = [rule_w[(q, s, w)] for w in range(6)]
        # Exactly one must be true
        # At least one
        clauses.append(w_vars[:])
        # At most one (pairwise negation)
        for i in range(6):
            for j in range(i + 1, 6):
                clauses.append([-w_vars[i], -w_vars[j]])

    # Bounce rule constraints
    # SR+7: new_q=1 (force), dir=L (force)
    clauses.append([rule_q[(0, 7)]])   # new_q = 1 (True)
    clauses.append([-rule_d[(0, 7)]])  # dir = L (False)
    # SL+0: new_q=0 (force), dir=R (force)
    clauses.append([-rule_q[(1, 0)]])  # new_q = 0 (False)
    clauses.append([rule_d[(1, 0)]])   # dir = R (True)

    # Trace variables for each scenario
    # For each scenario sid, time step t:
    #   tape[sid][t][x][s] = bool (one-hot over symbols 0..7 for position x)
    #   head[sid][t][x] = bool (one-hot over positions)
    #   state[sid][t] = bool (True=state 1, False=state 0)

    tape_v = {}   # (sid, t, x, s) -> var
    head_v = {}   # (sid, t, x) -> var
    state_v = {}  # (sid, t) -> var

    n_scenarios = len(scenarios)

    for sid in range(n_scenarios):
        for t in range(spc + 1):
            state_v[(sid, t)] = pool.new(f"st_{sid}_{t}")
            for x in range(eff_len):
                head_v[(sid, t, x)] = pool.new(f"h_{sid}_{t}_{x}")
                for s in range(8):
                    tape_v[(sid, t, x, s)] = pool.new(f"tp_{sid}_{t}_{x}_{s}")

                # One-hot for tape symbol at this position
                sym_vars = [tape_v[(sid, t, x, s)] for s in range(8)]
                clauses.append(sym_vars[:])
                for i in range(8):
                    for j in range(i + 1, 8):
                        clauses.append([-sym_vars[i], -sym_vars[j]])

            # One-hot for head position
            h_vars = [head_v[(sid, t, x)] for x in range(eff_len)]
            clauses.append(h_vars[:])
            for i in range(eff_len):
                for j in range(i + 1, eff_len):
                    clauses.append([-h_vars[i], -h_vars[j]])

    # Initial conditions for each scenario
    for sid in range(n_scenarios):
        init_tape = eff_tapes[sid]
        start_pos = starts[sid]

        # State = 0 (SR)
        clauses.append([-state_v[(sid, 0)]])
        # Head at start position
        for x in range(eff_len):
            if x == start_pos:
                clauses.append([head_v[(sid, 0, x)]])
            else:
                clauses.append([-head_v[(sid, 0, x)]])
        # Tape contents
        for x in range(eff_len):
            sym = init_tape[x]
            for s in range(8):
                if s == sym:
                    clauses.append([tape_v[(sid, 0, x, s)]])
                else:
                    clauses.append([-tape_v[(sid, 0, x, s)]])

    # Transition constraints for each scenario, each time step
    for sid in range(n_scenarios):
        for t in range(spc):
            # For each possible head position x:
            for x in range(eff_len):
                hv = head_v[(sid, t, x)]
                # For each possible state q:
                for q in range(2):
                    sv = state_v[(sid, t)] if q == 1 else -state_v[(sid, t)]
                    # For each possible read symbol s:
                    for s in range(8):
                        tv = tape_v[(sid, t, x, s)]

                        # Condition: head at x AND state=q AND tape[x]=s
                        # This means rule (q, s) fires.
                        # Implication: condition → effect

                        key = (q, s)
                        if key in fixed_rules:
                            nq, ns, nd = fixed_rules[key]
                            # Fixed rule: directly encode effects
                            new_x = x + nd
                            if new_x < 0 or new_x >= eff_len:
                                # This configuration leads to OOB → impossible
                                clauses.append([-hv, -sv if q == 1 else sv, -tv])
                                continue
                            _encode_fixed_effect(
                                clauses, sid, t, x, q, s, nq, ns, nd,
                                hv, sv, tv, tape_v, head_v, state_v,
                                eff_len, new_x
                            )
                        elif key in [(q2, s2) for (q2, s2) in all_search_rules]:
                            _encode_rule_effect(
                                clauses, sid, t, x, q, s,
                                hv, sv, tv, tape_v, head_v, state_v,
                                rule_q, rule_w, rule_d,
                                eff_len
                            )

    # Target conditions for each scenario
    for sid in range(n_scenarios):
        target_tape, target_head, target_state = scenarios[sid]

        # State at final step
        if target_state == 1:
            clauses.append([state_v[(sid, spc)]])
        else:
            clauses.append([-state_v[(sid, spc)]])

        # Head position at final step
        for x in range(eff_len):
            if x == target_head:
                clauses.append([head_v[(sid, spc, x)]])
            else:
                clauses.append([-head_v[(sid, spc, x)]])

        # Tape contents at final step
        for x in range(eff_len):
            sym = target_tape[x]
            for s in range(8):
                if s == sym:
                    clauses.append([tape_v[(sid, spc, x, s)]])
                else:
                    clauses.append([-tape_v[(sid, spc, x, s)]])

    return clauses, pool, rule_q, rule_w, rule_d, all_search_rules


def _encode_fixed_effect(clauses, sid, t, x, q, s, nq, ns, nd,
                         hv, sv, tv, tape_v, head_v, state_v,
                         eff_len, new_x):
    """Encode the effect of a fixed rule."""
    # Condition literals: hv (head at x), sv (state), tv (tape symbol)
    cond = [hv, sv if q == 1 else -state_v[(sid, t)], tv]
    # Wait, sv is already state_v or -state_v based on q
    # Let me be more careful
    cond_lits = [hv, tv]
    if q == 1:
        cond_lits.append(state_v[(sid, t)])
    else:
        cond_lits.append(-state_v[(sid, t)])

    neg_cond = [-l for l in cond_lits]

    # Effect on state at t+1
    if nq == 1:
        clauses.append(neg_cond + [state_v[(sid, t + 1)]])
    else:
        clauses.append(neg_cond + [-state_v[(sid, t + 1)]])

    # Effect on head at t+1
    clauses.append(neg_cond + [head_v[(sid, t + 1, new_x)]])

    # Effect on tape at t+1: position x gets ns, others unchanged
    for s2 in range(8):
        if s2 == ns:
            clauses.append(neg_cond + [tape_v[(sid, t + 1, x, s2)]])
        else:
            clauses.append(neg_cond + [-tape_v[(sid, t + 1, x, s2)]])

    # Other positions unchanged
    for x2 in range(eff_len):
        if x2 != x:
            for s2 in range(8):
                # tape[t+1][x2][s2] = tape[t][x2][s2]
                clauses.append(neg_cond + [-tape_v[(sid, t, x2, s2)],
                               tape_v[(sid, t + 1, x2, s2)]])
                clauses.append(neg_cond + [tape_v[(sid, t, x2, s2)],
                               -tape_v[(sid, t + 1, x2, s2)]])


def _encode_rule_effect(clauses, sid, t, x, q, s,
                        hv, sv, tv, tape_v, head_v, state_v,
                        rule_q, rule_w, rule_d, eff_len):
    """Encode the effect of a variable rule."""
    key = (q, s)
    cond_lits = [hv, tv]
    if q == 1:
        cond_lits.append(state_v[(sid, t)])
    else:
        cond_lits.append(-state_v[(sid, t)])

    neg_cond = [-l for l in cond_lits]

    # Effect on state at t+1: depends on rule_q[key]
    # If rule_q[key] is True (new_q=1), then state[t+1] must be 1
    # cond ∧ rule_q → state[t+1]
    # cond ∧ ¬rule_q → ¬state[t+1]
    clauses.append(neg_cond + [-rule_q[key], state_v[(sid, t + 1)]])
    clauses.append(neg_cond + [rule_q[key], -state_v[(sid, t + 1)]])

    # Effect on head at t+1: depends on rule_d[key]
    # If rule_d[key] is True (dir=R, +1), head moves to x+1
    # If rule_d[key] is False (dir=L, -1), head moves to x-1
    new_x_r = x + 1
    new_x_l = x - 1

    if new_x_r >= eff_len and new_x_l < 0:
        # Both directions OOB → impossible
        clauses.append(neg_cond + [])  # Empty clause under condition = UNSAT if condition met
        # Actually we need: condition → False, i.e., ¬condition
        clauses.append([-l for l in cond_lits])
        return
    elif new_x_r >= eff_len:
        # Right OOB → must go left
        clauses.append(neg_cond + [-rule_d[key]])  # Force dir=L
        clauses.append(neg_cond + [head_v[(sid, t + 1, new_x_l)]])
    elif new_x_l < 0:
        # Left OOB → must go right
        clauses.append(neg_cond + [rule_d[key]])  # Force dir=R
        clauses.append(neg_cond + [head_v[(sid, t + 1, new_x_r)]])
    else:
        # Both directions valid
        # dir=R → head at x+1
        clauses.append(neg_cond + [-rule_d[key], head_v[(sid, t + 1, new_x_r)]])
        # dir=L → head at x-1
        clauses.append(neg_cond + [rule_d[key], head_v[(sid, t + 1, new_x_l)]])

    # Effect on tape at position x: write symbol
    for w in range(6):
        ws = w + 1  # Actual symbol 1..6
        for s2 in range(8):
            if s2 == ws:
                # rule_w[key, w] → tape[t+1][x][s2]
                clauses.append(neg_cond + [-rule_w[(key[0], key[1], w)],
                               tape_v[(sid, t + 1, x, s2)]])
            else:
                # rule_w[key, w] → ¬tape[t+1][x][s2]
                clauses.append(neg_cond + [-rule_w[(key[0], key[1], w)],
                               -tape_v[(sid, t + 1, x, s2)]])

    # Other positions unchanged
    for x2 in range(eff_len):
        if x2 != x:
            for s2 in range(8):
                clauses.append(neg_cond + [-tape_v[(sid, t, x2, s2)],
                               tape_v[(sid, t + 1, x2, s2)]])
                clauses.append(neg_cond + [tape_v[(sid, t, x2, s2)],
                               -tape_v[(sid, t + 1, x2, s2)]])


def decode_solution(model, rule_q, rule_w, rule_d, all_search_rules):
    """Extract transition table from SAT model."""
    table = {
        (0, 0): (0, 0, 1),
        (1, 7): (1, 7, -1),
    }
    for (q, s) in all_search_rules:
        nq = 1 if model[rule_q[(q, s)] - 1] > 0 else 0
        nd = 1 if model[rule_d[(q, s)] - 1] > 0 else -1
        ws = None
        for w in range(6):
            if model[rule_w[(q, s, w)] - 1] > 0:
                ws = w + 1
                break
        if ws is None:
            ws = 1  # Fallback
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
    """Verify on larger tape."""
    print("\n--- Full tape verification ---")
    xm = 20
    lp = 8
    all_ok = True
    for i in range(len(payloads) - 1):
        pi = payloads[i]
        po = payloads[i + 1]
        tape = [0] * lp + list(pi) + [7] * (xm - lp - len(pi))
        h, s = lp, 0
        found = False
        for t in range(1, spc * 3 + 1):
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
                actual = tape[lp:lp + len(po)]
                if actual == list(po):
                    ok_boundary = (all(tape[x] == 0 for x in range(lp)) and
                                   all(tape[x] == 7 for x in range(lp + len(po), xm)))
                    print(f"  [{'OK' if ok_boundary else 'PARTIAL'}] {pi} → {actual} "
                          f"at t={t} boundary={'clean' if ok_boundary else 'dirty'}")
                    found = True
                    break
        if not found:
            print(f"  [FAIL] {pi} → expected {po}")
            all_ok = False

    if all_ok:
        print("\n*** FULL TAPE VERIFIED ***")
        # Generalization
        for a_val in [1, 2, 3, 4, 5, 8]:
            xb = 80
            lp = 25
            pay = [1] * a_val + [3]
            tb = [0] * lp + pay + [7] * (xb - lp - len(pay))
            max_s = spc * (2 * a_val + 2) * 5
            hb, sb = lp, 0
            cyc = False
            for t in range(1, max_s + 1):
                if hb < 0 or hb >= xb:
                    break
                key = (sb, tb[hb])
                if key not in table:
                    break
                nq, ns, nd = table[key]
                tb[hb] = ns
                sb = nq
                hb += nd
                if hb == lp and sb == 0 and t > spc:
                    p = tb[lp:lp + a_val + 1]
                    if p == [1] * a_val + [3]:
                        l = 0
                        while l < xb and tb[l] == 0:
                            l += 1
                        if l == lp:
                            print(f"  A={a_val}: CYCLE at t={t}")
                            cyc = True
                            break
            if not cyc:
                l = 0
                while l < xb and tb[l] == 0:
                    l += 1
                r = xb - 1
                while r >= 0 and tb[r] == 7:
                    r -= 1
                print(f"  A={a_val}: NO CYCLE. pay={tb[l:r + 1][:12]}")
    return all_ok


def search(spc=8, el=1, er=0, solver_name='cadical153'):
    """Main search."""
    payloads = [
        [1, 1, 3],
        [1, 3, 4],
        [3, 4, 4],
        [5, 4, 4],
        [1, 5, 4],
        [1, 1, 5],
        [1, 1, 3],  # cycle close
    ]
    descs = [
        'I0: dec A (A=2→1)',
        'I0: dec A (A=1→0)',
        'I0→I1: switch',
        'I1: dec B (B=2→1)',
        'I1: dec B (B=1→0)',
        'I1→I0: switch',
    ]

    left_b = [0] * el
    right_b = [7] * er
    start = el
    eff_len = el + 3 + er

    eff_tapes = [left_b + p + right_b for p in payloads]
    scenarios = [(eff_tapes[i + 1], start, 0) for i in range(6)]
    starts = [start] * 7

    eff_desc = f"{el}L+3P+{er}R (eff_len={eff_len})"
    print(f"\n{'=' * 60}")
    print(f"SAT Old encoding: spc={spc} eff={eff_desc}")
    print(f"{'=' * 60}")
    for i, d in enumerate(descs):
        print(f"  {i}: {payloads[i]} → {payloads[i + 1]}  ({d})")

    print(f"\nEncoding to SAT...")
    t0 = time.time()
    clauses, pool, rule_q, rule_w, rule_d, all_search_rules = \
        encode_holographic_sat(scenarios, eff_tapes, starts, spc, eff_len)

    n_vars = pool.next_var - 1
    n_clauses = len(clauses)
    elapsed_enc = time.time() - t0
    print(f"  Variables: {n_vars:,}")
    print(f"  Clauses: {n_clauses:,}")
    print(f"  Encoding time: {elapsed_enc:.1f}s")

    print(f"\nSolving with {solver_name}...")
    t1 = time.time()
    solver = SATSolver(name=solver_name, bootstrap_with=clauses)
    result = solver.solve()
    elapsed_solve = time.time() - t1
    print(f"  Result: {'SAT' if result else 'UNSAT'} ({elapsed_solve:.1f}s)")

    if result:
        model = solver.get_model()
        table = decode_solution(model, rule_q, rule_w, rule_d, all_search_rules)
        print_table(table)
        verify_full(table, payloads, spc)

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open(f"solution_sat_spc{spc}.json", "w") as f:
            json.dump(sol, f, indent=2)
        print(f"\nSaved to solution_sat_spc{spc}.json")
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
    p.add_argument("--solver", default="cadical153",
                   help="SAT solver: cadical153, glucose4, minisat22, etc.")
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1):
        for el, er in [(args.el, args.er)]:
            table = search(spc=spc, el=el, er=er, solver_name=args.solver)
            if table:
                sys.exit(0)
