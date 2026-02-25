#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Rule 110 SAT Search v2

More efficient approach: encode a SINGLE sweep over a wider tape,
not individual test cases. One sweep should transform the entire
Rule 110 configuration.

Key insight: if the TM can correctly update ONE 5-cell configuration
(checking only the interior 3 cells), then the same rules will work
for arbitrary width.

We use incrementally larger step counts and add test cases iteratively.
"""

import time
import sys
import itertools
from pysat.solvers import Solver as SATSolver


# Rule 110 lookup
RULE110 = {}
for i in range(8):
    L = (i >> 2) & 1
    C = (i >> 1) & 1
    R = i & 1
    RULE110[(L, C, R)] = (110 >> i) & 1

def rule110_step(config, left_bc=0, right_bc=0):
    n = len(config)
    result = []
    for i in range(n):
        L = config[i-1] if i > 0 else left_bc
        C = config[i]
        R = config[i+1] if i < n-1 else right_bc
        result.append(RULE110[(L, C, R)])
    return result


class VarPool:
    def __init__(self):
        self.next_var = 1
    def new(self):
        v = self.next_var
        self.next_var += 1
        return v


def encode_single_case(pool, clauses, rule_q, rule_w, rule_d, search_rules,
                        fixed, input_config, output_config, spc, el, er,
                        check_interior_only=False):
    """Encode one test case. Returns clause count added."""
    pay_len = len(input_config)
    eff_len = el + pay_len + er
    init_tape = [0] * el + [c + 1 for c in input_config] + [7] * er
    expected_payload = [c + 1 for c in output_config]
    start_pos = el

    n0 = len(clauses)
    tape_v = {}; head_v = {}; state_v = {}

    for t in range(spc + 1):
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
        if x == start_pos:
            clauses.append([head_v[(0, x)]])
        else:
            clauses.append([-head_v[(0, x)]])
        for s in range(8):
            if s == init_tape[x]:
                clauses.append([tape_v[(0, x, s)]])
            else:
                clauses.append([-tape_v[(0, x, s)]])

    # Transitions
    for t in range(spc):
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
                            clauses.append(neg)
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
                        clauses.append(neg + [-rule_q[key], state_v[t+1]])
                        clauses.append(neg + [rule_q[key], -state_v[t+1]])
                        nxr = x + 1; nxl = x - 1
                        if nxr >= eff_len and nxl < 0:
                            clauses.append(neg)
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
                        for w in range(6):
                            ws = w + 1
                            for s2 in range(8):
                                if s2 == ws: clauses.append(neg + [-rule_w[(key[0],key[1],w)], tape_v[(t+1, x, s2)]])
                                else: clauses.append(neg + [-rule_w[(key[0],key[1],w)], -tape_v[(t+1, x, s2)]])
                        for x2 in range(eff_len):
                            if x2 != x:
                                for s2 in range(8):
                                    clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t+1, x2, s2)]])
                                    clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t+1, x2, s2)]])

    # Final conditions
    clauses.append([-state_v[spc]])
    clauses.append([head_v[(spc, start_pos)]])

    if check_interior_only:
        # Only check interior cells (skip first and last)
        for j in range(1, pay_len - 1):
            for s in range(8):
                if s == expected_payload[j]:
                    clauses.append([tape_v[(spc, start_pos + j, s)]])
                else:
                    clauses.append([-tape_v[(spc, start_pos + j, s)]])
    else:
        for j, sym in enumerate(expected_payload):
            for s in range(8):
                if s == sym:
                    clauses.append([tape_v[(spc, start_pos + j, s)]])
                else:
                    clauses.append([-tape_v[(spc, start_pos + j, s)]])

    # Clean boundaries
    for x in range(el):
        clauses.append([tape_v[(spc, x, 0)]])
    for x in range(el + pay_len, eff_len):
        clauses.append([tape_v[(spc, x, 7)]])

    return len(clauses) - n0


def setup_rules(pool, clauses):
    """Create and constrain rule variables."""
    fixed = {(0, 0): (0, 0, 1), (1, 7): (1, 7, -1)}
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

    return fixed, rule_q, rule_w, rule_d, search_rules


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
    print(f"\n  {'St':>2}  {'Sym':>3} ->  {'Wr':>2}  {'Dir':>3}  {'Nxt':>3}")
    print("-" * 30)
    for q in range(2):
        for s in range(8):
            if (q, s) in table:
                qo, so, do = table[(q, s)]
                print(f"  {sn[q]:>2}  {s:>3} ->  {so:>2}  {dn[do]:>3}  {sn[qo]:>3}")


def simulate(table, tape_init, head, state, max_steps=2000):
    """Simulate TM and return (tape, head, state, steps)."""
    tape = list(tape_init)
    h, s = head, state
    xm = len(tape)
    for t in range(1, max_steps + 1):
        if h < 0 or h >= xm:
            return tape, h, s, t
        key = (s, tape[h])
        if key not in table:
            return tape, h, s, t
        nq, ns, nd = table[key]
        tape[h] = ns; s = nq; h += nd
        if h == head and s == 0:
            return tape, h, s, t
    return tape, h, s, max_steps


def verify_rule110(table, input_config, expected_output, el=10, max_steps=2000):
    """Verify one Rule 110 step."""
    pay_len = len(input_config)
    xm = el + pay_len + el
    tape = [0] * el + [c + 1 for c in input_config] + [7] * el
    tape_out, h, s, steps = simulate(table, tape, el, 0, max_steps)
    if h != el or s != 0:
        return False, steps
    actual = tape_out[el:el + pay_len]
    expected = [c + 1 for c in expected_output]
    return actual == expected, steps


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=3)
    p.add_argument("--spc", type=int, default=6)
    p.add_argument("--spc-max", type=int, default=30)
    p.add_argument("--el", type=int, default=2)
    p.add_argument("--er", type=int, default=2)
    p.add_argument("--interior", action="store_true",
                   help="Only check interior cells")
    args = p.parse_args()

    # Generate ALL test cases for this cell count
    all_cases = []
    for bits in itertools.product([0, 1], repeat=args.cells):
        inp = list(bits)
        out = rule110_step(inp, left_bc=0, right_bc=0)
        all_cases.append((inp, out))

    print(f"Rule 110 ({args.cells} cells, {len(all_cases)} cases)")
    for i, (inp, out) in enumerate(all_cases):
        print(f"  {i}: {inp} -> {out}")

    for spc in range(args.spc, args.spc_max + 1):
        eff_len = args.el + args.cells + args.er
        print(f"\n{'='*60}")
        print(f"Rule 110 SAT: cells={args.cells} spc={spc} eff={eff_len}")
        print(f"{'='*60}")
        sys.stdout.flush()

        pool = VarPool()
        clauses = []

        t0 = time.time()
        fixed, rq, rw, rd, sr = setup_rules(pool, clauses)

        for i, (inp, out) in enumerate(all_cases):
            encode_single_case(pool, clauses, rq, rw, rd, sr, fixed,
                              inp, out, spc, args.el, args.er,
                              check_interior_only=args.interior)

        enc_time = time.time() - t0
        print(f"  Vars: {pool.next_var - 1:,}  Clauses: {len(clauses):,}")
        print(f"  Encoding: {enc_time:.1f}s")
        sys.stdout.flush()

        t1 = time.time()
        solver = SATSolver(name='cadical153', bootstrap_with=clauses)
        result = solver.solve()
        solve_time = time.time() - t1
        print(f"  Result: {'SAT' if result else 'UNSAT'} ({solve_time:.1f}s)")
        sys.stdout.flush()

        if result:
            model = solver.get_model()
            table = decode_solution(model, rq, rw, rd, sr)
            print_table(table)

            # Full verification
            print(f"\n--- Verification (all {len(all_cases)} cases) ---")
            all_ok = True
            for i, (inp, out) in enumerate(all_cases):
                ok, steps = verify_rule110(table, inp, out)
                print(f"  {'[OK]' if ok else '[FAIL]'} {inp} -> {out} t={steps}")
                if not ok: all_ok = False

            if all_ok:
                print(f"\n*** ALL {len(all_cases)} CASES VERIFIED ***")
                # Generalization
                print(f"\n--- Generalization ---")
                gen_ok = True
                for n in [args.cells + 1, args.cells + 2, 6, 8, 10]:
                    if n <= args.cells:
                        continue
                    gen_cases = []
                    for bits in itertools.product([0, 1], repeat=n):
                        gen_cases.append((list(bits), rule110_step(list(bits))))
                        if len(gen_cases) >= 32:
                            break
                    ok_cnt = sum(1 for inp, out in gen_cases
                                if verify_rule110(table, inp, out, max_steps=5000)[0])
                    tag = "PASS" if ok_cnt == len(gen_cases) else "FAIL"
                    print(f"  [{tag}] {n} cells: {ok_cnt}/{len(gen_cases)}")
                    if ok_cnt < len(gen_cases):
                        gen_ok = False

                if gen_ok:
                    print(f"\n*** GENERALIZES! Universal Rule 110 simulator found! ***")
                else:
                    print(f"\n  Does not generalize. Trying next spc...")
                    solver.delete()
                    continue

            solver.delete()
            sys.exit(0)

        solver.delete()
