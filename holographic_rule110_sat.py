#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Rule 110 SAT Search

Find a 2-state, 8-symbol transition table that simulates Rule 110
cellular automaton. The TM starts with a Rule 110 configuration
encoded on tape, runs for K steps, and produces the next generation.

Approach: SAT-based search using chained test cases.
"""

import time
import sys
import itertools
from pysat.solvers import Solver as SATSolver


# Rule 110 lookup: (left, center, right) -> new_center
RULE110 = {}
for i in range(8):
    L = (i >> 2) & 1
    C = (i >> 1) & 1
    R = i & 1
    # Rule 110 = 01101110 in binary
    RULE110[(L, C, R)] = (110 >> i) & 1

def rule110_step(config, left_bc=0, right_bc=0):
    """Compute one step of Rule 110 on a finite configuration."""
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


def encode_rule110_sat(test_cases, spc, el, er):
    """
    Encode SAT problem: find transition table such that for each test case,
    the TM transforms the input configuration to the output configuration
    in exactly spc steps.

    test_cases: list of (input_config, output_config) pairs
                each config is a list of Rule 110 cell values (0 or 1)

    Tape encoding: symbol 1 = CA state 0, symbol 2 = CA state 1
    The TM starts at the leftmost payload cell in state SR (0).
    After spc steps, should be at the leftmost payload cell in state SR (0)
    with the output configuration on tape.
    """
    pool = VarPool()
    clauses = []

    # Fixed boundary rules
    fixed = {
        (0, 0): (0, 0, 1),   # SR+0 -> SR, write 0, move R
        (1, 7): (1, 7, -1),  # SL+7 -> SL, write 7, move L
    }

    # Rule variables: the 14 search rules
    rule_q = {}   # new state (bool: true=1/SL, false=0/SR)
    rule_w = {}   # write symbol (one-hot over {1,2,3,4,5,6})
    rule_d = {}   # direction (bool: true=R/+1, false=L/-1)

    search_rules = [(q, s) for q in range(2) for s in range(1, 7)]
    search_rules += [(0, 7), (1, 0)]

    for (q, s) in search_rules:
        rule_q[(q, s)] = pool.new()
        for w in range(6):
            rule_w[(q, s, w)] = pool.new()
        rule_d[(q, s)] = pool.new()

        # One-hot constraint on write symbol
        w_vars = [rule_w[(q, s, w)] for w in range(6)]
        clauses.append(w_vars[:])  # at least one
        for i in range(6):
            for j in range(i + 1, 6):
                clauses.append([-w_vars[i], -w_vars[j]])  # at most one

    # Bounce constraints
    clauses.append([rule_q[(0, 7)]])    # SR+7: must go to SL
    clauses.append([-rule_d[(0, 7)]])   # SR+7: must move L
    clauses.append([-rule_q[(1, 0)]])   # SL+0: must go to SR
    clauses.append([rule_d[(1, 0)]])    # SL+0: must move R

    def encode_test_case(tc_id, input_config, output_config):
        """Encode one test case as SAT constraints."""
        pay_len = len(input_config)
        eff_len = el + pay_len + er

        # Build initial tape
        init_tape = [0] * el + [c + 1 for c in input_config] + [7] * er

        # Build expected output tape (just payload region)
        expected_payload = [c + 1 for c in output_config]

        tape_v = {}
        head_v = {}
        state_v = {}

        for t in range(spc + 1):
            state_v[t] = pool.new()
            for x in range(eff_len):
                head_v[(t, x)] = pool.new()
                for s in range(8):
                    tape_v[(t, x, s)] = pool.new()
                # One-hot on tape symbols
                sym_vars = [tape_v[(t, x, s)] for s in range(8)]
                clauses.append(sym_vars[:])
                for i in range(8):
                    for j in range(i + 1, 8):
                        clauses.append([-sym_vars[i], -sym_vars[j]])
            # One-hot on head position
            h_vars = [head_v[(t, x)] for x in range(eff_len)]
            clauses.append(h_vars[:])
            for i in range(eff_len):
                for j in range(i + 1, eff_len):
                    clauses.append([-h_vars[i], -h_vars[j]])

        # Initial conditions: t=0
        clauses.append([-state_v[0]])  # state = SR (0)
        start_pos = el  # head starts at leftmost payload cell
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

        # Transition constraints
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
                            if nq == 1:
                                clauses.append(neg + [state_v[t+1]])
                            else:
                                clauses.append(neg + [-state_v[t+1]])
                            clauses.append(neg + [head_v[(t+1, nx)]])
                            for s2 in range(8):
                                if s2 == ns:
                                    clauses.append(neg + [tape_v[(t+1, x, s2)]])
                                else:
                                    clauses.append(neg + [-tape_v[(t+1, x, s2)]])
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t+1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t+1, x2, s2)]])

                        elif key in [(qq,ss) for (qq,ss) in search_rules]:
                            # State
                            clauses.append(neg + [-rule_q[key], state_v[t+1]])
                            clauses.append(neg + [rule_q[key], -state_v[t+1]])
                            # Head movement
                            nxr = x + 1
                            nxl = x - 1
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
                            # Write symbol
                            for w in range(6):
                                ws = w + 1
                                for s2 in range(8):
                                    if s2 == ws:
                                        clauses.append(neg + [-rule_w[(key[0], key[1], w)], tape_v[(t+1, x, s2)]])
                                    else:
                                        clauses.append(neg + [-rule_w[(key[0], key[1], w)], -tape_v[(t+1, x, s2)]])
                            # Frame: unchanged cells
                            for x2 in range(eff_len):
                                if x2 != x:
                                    for s2 in range(8):
                                        clauses.append(neg + [-tape_v[(t, x2, s2)], tape_v[(t+1, x2, s2)]])
                                        clauses.append(neg + [tape_v[(t, x2, s2)], -tape_v[(t+1, x2, s2)]])

        # Final conditions: t=spc
        # State should be SR (0), head at start position
        clauses.append([-state_v[spc]])  # state = SR
        clauses.append([head_v[(spc, start_pos)]])  # head at start

        # Payload should match expected output
        for j, sym in enumerate(expected_payload):
            for s in range(8):
                if s == sym:
                    clauses.append([tape_v[(spc, start_pos + j, s)]])
                else:
                    clauses.append([-tape_v[(spc, start_pos + j, s)]])

        # Boundaries should be clean
        for x in range(el):
            clauses.append([tape_v[(spc, x, 0)]])
        for x in range(el + pay_len, eff_len):
            clauses.append([tape_v[(spc, x, 7)]])

    # Encode all test cases
    for i, (inp, out) in enumerate(test_cases):
        encode_test_case(i, inp, out)

    return clauses, pool, rule_q, rule_w, rule_d, search_rules


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


def verify_rule110(table, input_config, expected_output, max_steps=500):
    """Verify that the TM correctly transforms input to output."""
    xm = 60
    lp = 20  # left padding
    pay_len = len(input_config)

    tape = [0] * lp + [c + 1 for c in input_config] + [7] * (xm - lp - pay_len)
    h, s = lp, 0

    for t in range(1, max_steps + 1):
        if h < 0 or h >= xm:
            return False, t, None
        key = (s, tape[h])
        if key not in table:
            return False, t, None
        nq, ns, nd = table[key]
        tape[h] = ns
        s = nq
        h += nd

        # Check if we're back at start position in state SR
        if h == lp and s == 0:
            actual = tape[lp:lp + pay_len]
            expected = [c + 1 for c in expected_output]
            if actual == expected:
                # Check boundaries
                lb = all(tape[x] == 0 for x in range(lp))
                rb = all(tape[x] == 7 for x in range(lp + pay_len, xm))
                return True, t, "OK" if lb and rb else "DIRTY"
    return False, max_steps, None


def generate_test_cases(n_cells, n_cases=None):
    """Generate Rule 110 test cases with n_cells cells."""
    cases = []
    if n_cases is None:
        # All possible configurations
        for bits in itertools.product([0, 1], repeat=n_cells):
            inp = list(bits)
            out = rule110_step(inp, left_bc=0, right_bc=0)
            cases.append((inp, out))
    else:
        # Random subset
        import random
        seen = set()
        while len(cases) < n_cases:
            bits = tuple(random.randint(0, 1) for _ in range(n_cells))
            if bits not in seen:
                seen.add(bits)
                inp = list(bits)
                out = rule110_step(inp, left_bc=0, right_bc=0)
                cases.append((inp, out))
    return cases


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=3, help="Number of CA cells")
    p.add_argument("--spc", type=int, default=8, help="Steps per CA step (min)")
    p.add_argument("--spc-max", type=int, default=20, help="Steps per CA step (max)")
    p.add_argument("--el", type=int, default=2, help="Left boundary padding")
    p.add_argument("--er", type=int, default=2, help="Right boundary padding")
    p.add_argument("--cases", type=int, default=0, help="Max test cases (0=all)")
    args = p.parse_args()

    # Generate test cases
    n_cases = None if args.cases == 0 else args.cases
    test_cases = generate_test_cases(args.cells, n_cases)

    print(f"Rule 110 test cases ({args.cells} cells, {len(test_cases)} cases):")
    for i, (inp, out) in enumerate(test_cases):
        print(f"  {i}: {inp} -> {out}")

    for spc in range(args.spc, args.spc_max + 1):
        eff_len = args.el + args.cells + args.er
        print(f"\n{'='*60}")
        print(f"Rule 110 SAT: {args.cells} cells, spc={spc}, eff_len={eff_len}")
        print(f"  {args.el}L + {args.cells}P + {args.er}R = {eff_len} tape cells")
        print(f"  {len(test_cases)} test cases")
        print(f"{'='*60}")

        t0 = time.time()
        clauses, pool, rq, rw, rd, sr = encode_rule110_sat(
            test_cases, spc, args.el, args.er)
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
            print_table(table)

            print(f"\n--- Verification ---")
            all_ok = True
            for i, (inp, out) in enumerate(test_cases):
                ok, steps, tag = verify_rule110(table, inp, out)
                status = f"[{tag}]" if ok else "[FAIL]"
                print(f"  {status} Case {i}: {inp} -> {out} (t={steps})")
                if not ok:
                    all_ok = False

            if all_ok:
                print(f"\n*** ALL {len(test_cases)} CASES VERIFIED ***")

                # Test generalization: try larger configurations
                print(f"\n--- Generalization test ---")
                for n in [args.cells + 1, args.cells + 2, 6, 8]:
                    if n <= args.cells:
                        continue
                    extra_cases = generate_test_cases(n, n_cases=min(16, 2**n))
                    ok_count = 0
                    for inp, out in extra_cases:
                        ok, _, _ = verify_rule110(table, inp, out, max_steps=2000)
                        if ok:
                            ok_count += 1
                    print(f"  {n} cells: {ok_count}/{len(extra_cases)} correct")

            solver.delete()
            sys.exit(0)

        solver.delete()
