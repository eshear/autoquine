#!/usr/bin/env python3
"""
(2, 8) Holographic TM — DFS Search with Constraint Propagation

Instead of Z3 (which times out) or SA (which gets stuck at local optima),
use depth-first search with constraint propagation.

Key insight: simulate step-by-step, and when an undefined rule is needed,
branch over all valid assignments. Once a rule is fixed, it applies everywhere.
This gives much tighter pruning than Z3's generic SMT solver.
"""

import time
import json
import sys
from copy import deepcopy


# Fixed boundary rules
FIXED = {
    (0, 0): (0, 0, 1),    # SR+0: pass through left
    (1, 7): (1, 7, -1),   # SL+7: pass through right
}

# Constrained bounce rules (state and direction fixed, write symbol free)
# SR+7 → (1, W, -1) where W ∈ {1..6}
# SL+0 → (0, V, 1) where V ∈ {1..6}

def valid_assignments(q, s):
    """Return all valid (new_q, write_s, dir) for rule (q, s)."""
    if (q, s) in FIXED:
        return [FIXED[(q, s)]]
    if q == 0 and s == 7:  # SR+7: bounce
        return [(1, w, -1) for w in range(1, 7)]
    if q == 1 and s == 0:  # SL+0: bounce
        return [(0, v, 1) for v in range(1, 7)]
    # Payload rules: state ∈ {0,1}, write ∈ {1..6}, dir ∈ {-1, 1}
    if 1 <= s <= 6:
        return [(nq, ns, d) for nq in range(2)
                for ns in range(1, 7) for d in (-1, 1)]
    return []


def simulate_step(tape, head, state, table, eff_len):
    """Simulate one step. Returns (tape, head, state, rule_key) or None if OOB."""
    if head < 0 or head >= eff_len:
        return None
    sym = tape[head]
    key = (state, sym)
    if key not in table:
        return key  # Need to assign this rule
    nq, ns, nd = table[key]
    tape = list(tape)
    tape[head] = ns
    return (tape, head + nd, nq, key)


def simulate_scenario(table, eff_tape, start, spc, target_tape, target_head, target_state):
    """Simulate spc steps. Returns True if target reached, False if failed,
    or (state, symbol) key if we need a new rule assignment."""
    tape = list(eff_tape)
    head = start
    state = 0
    eff_len = len(tape)

    for t in range(spc):
        if head < 0 or head >= eff_len:
            return False  # Out of bounds
        sym = tape[head]
        key = (state, sym)
        if key not in table:
            return None  # Shouldn't happen in full simulation
        nq, ns, nd = table[key]
        tape[head] = ns
        state = nq
        head += nd

    # Check target
    if head != target_head or state != target_state:
        return False
    for i, s in enumerate(target_tape):
        if tape[i] != s:
            return False
    return True


def dfs_search(scenarios, eff_tapes, starts, spc, callback=None):
    """
    DFS search for transition table satisfying all scenarios.

    Each scenario: (eff_tape_in, target_tape, target_head, target_state)
    """
    best = [0]  # Best number of scenarios satisfied
    calls = [0]
    t0 = time.time()
    last_report = [t0]

    def try_scenario(table, scenario_idx, tape, head, state, step, spc_val):
        """Try to complete one scenario from current state."""
        calls[0] += 1
        eff_len = len(tape)

        if calls[0] % 500000 == 0:
            elapsed = time.time() - t0
            rate = calls[0] / elapsed if elapsed > 0 else 0
            print(f"  DFS: {calls[0]:,} calls, {rate:.0f}/s, "
                  f"best={best[0]}/{len(scenarios)}, "
                  f"current scenario={scenario_idx}, step={step}")

        # Simulate remaining steps
        for t in range(step, spc_val):
            if head < 0 or head >= eff_len:
                return None  # Out of bounds, fail
            sym = tape[head]
            key = (state, sym)

            if key not in table:
                # Branch over all valid assignments
                assignments = valid_assignments(key[0], key[1])
                for nq, ns, nd in assignments:
                    new_head = head + nd
                    if new_head < 0 or new_head >= eff_len:
                        continue  # Skip OOB moves
                    new_tape = list(tape)
                    new_tape[head] = ns
                    new_table = dict(table)
                    new_table[key] = (nq, ns, nd)
                    result = try_scenario(
                        new_table, scenario_idx,
                        new_tape, new_head, nq, t + 1, spc_val
                    )
                    if result is not None:
                        return result
                return None  # No valid assignment found

            nq, ns, nd = table[key]
            tape = list(tape)
            tape[head] = ns
            state = nq
            head += nd

        # Check target
        target_tape, target_head, target_state = scenarios[scenario_idx]
        if head != target_head or state != target_state:
            return None
        target_eff = list(eff_tapes[scenario_idx + 1]) if scenario_idx + 1 < len(eff_tapes) else list(target_tape)
        for i in range(len(tape)):
            if tape[i] != target_eff[i]:
                return None

        # Scenario satisfied!
        if scenario_idx + 1 > best[0]:
            best[0] = scenario_idx + 1
            elapsed = time.time() - t0
            print(f"  *** Scenario {scenario_idx} OK ({best[0]}/{len(scenarios)}) "
                  f"at {calls[0]:,} calls ({elapsed:.1f}s)")
            if callback:
                callback(table, scenario_idx + 1)

        if scenario_idx + 1 >= len(scenarios):
            return table  # All scenarios done!

        # Try next scenario
        next_tape = list(eff_tapes[scenario_idx + 1])
        next_start = starts[scenario_idx + 1]
        return try_scenario(
            table, scenario_idx + 1,
            next_tape, next_start, 0, 0, spc_val
        )

    # Start with fixed rules only
    init_table = dict(FIXED)
    first_tape = list(eff_tapes[0])
    first_start = starts[0]

    return try_scenario(init_table, 0, first_tape, first_start, 0, 0, spc)


def search_old_encoding(spc=8, el=1, er=0):
    """Search for old encoding A=2 cycle."""
    # Old encoding A=2 cycle
    payloads = [
        [1, 1, 3],  # A=2, I0
        [1, 3, 4],  # A=1, I0, B=1
        [3, 4, 4],  # A=0, I0, B=2
        [5, 4, 4],  # A=0, I1, B=2
        [1, 5, 4],  # A=1, I1, B=1
        [1, 1, 5],  # A=1, I1+, B=0
        [1, 1, 3],  # Back to start (cycle close)
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

    # Build effective tapes for each checkpoint
    eff_tapes = [left_b + p + right_b for p in payloads]

    # Scenarios: (target_tape, target_head, target_state)
    scenarios = [(eff_tapes[i + 1], start, 0) for i in range(6)]
    starts = [start] * 7

    eff_desc = f"{el}L+3P+{er}R"
    print(f"\n{'='*60}")
    print(f"DFS Old encoding: spc={spc} eff={eff_desc}")
    print(f"{'='*60}")
    for i, d in enumerate(descs):
        print(f"  {i}: {payloads[i]} → {payloads[i+1]}  ({d})")

    t0 = time.time()

    def on_progress(table, n_done):
        elapsed = time.time() - t0
        print(f"    Table so far ({n_done} scenarios): "
              f"{sum(1 for k in table if k not in FIXED)} payload rules defined")

    result = dfs_search(scenarios, eff_tapes, starts, spc, callback=on_progress)

    elapsed = time.time() - t0
    if result:
        print(f"\n*** FOUND in {elapsed:.1f}s ***")
        print_table(result)
        verify_full(result, payloads, spc)
        return result
    else:
        print(f"\nNo solution found ({elapsed:.1f}s)")
        return None


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
    """Verify on a larger tape."""
    print("\n--- Full tape verification ---")
    xm = 20; lp = 8
    all_ok = True
    for i in range(len(payloads) - 1):
        pi = payloads[i]
        po = payloads[i + 1]
        tape = [0]*lp + list(pi) + [7]*(xm - lp - len(pi))
        h, s = lp, 0
        found = False
        for t in range(1, spc * 3 + 1):
            if h < 0 or h >= xm:
                break
            key = (s, tape[h])
            if key not in table:
                print(f"  [MISSING RULE] {key}")
                break
            nq, ns, nd = table[key]
            tape[h] = ns; s = nq; h += nd
            if h == lp and s == 0:
                actual = tape[lp:lp + len(po)]
                if actual == list(po):
                    print(f"  [OK] {pi} → {actual} at t={t}")
                    found = True
                    break
        if not found:
            print(f"  [FAIL] {pi} → expected {po}")
            all_ok = False

    if all_ok:
        print("\n*** FULL TAPE VERIFIED ***")
        # Generalization test
        for a_val in [1, 2, 3, 4, 5]:
            xb = 60; lp = 20
            pay = [1]*a_val + [3]
            tb = [0]*lp + pay + [7]*(xb - lp - len(pay))
            max_s = spc * (2*a_val + 2) * 5
            hb, sb = lp, 0
            cyc = False
            for t in range(1, max_s + 1):
                if hb < 0 or hb >= xb:
                    break
                key = (sb, tb[hb])
                if key not in table:
                    break
                nq, ns, nd = table[key]
                tb[hb] = ns; sb = nq; hb += nd
                if hb == lp and sb == 0 and t > spc:
                    p = tb[lp:lp + a_val + 1]
                    if p == [1]*a_val + [3]:
                        l = 0
                        while l < xb and tb[l] == 0: l += 1
                        if l == lp:
                            print(f"  A={a_val}: CYCLE at t={t}")
                            cyc = True
                            break
            if not cyc:
                l = 0
                while l < xb and tb[l] == 0: l += 1
                r = xb - 1
                while r >= 0 and tb[r] == 7: r -= 1
                print(f"  A={a_val}: NO CYCLE. pay={tb[l:r+1][:12]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spc", type=int, default=6)
    p.add_argument("--spc-max", type=int, default=12)
    p.add_argument("--el", type=int, default=1)
    p.add_argument("--er", type=int, default=0)
    args = p.parse_args()

    for spc in range(args.spc, args.spc_max + 1):
        table = search_old_encoding(spc=spc, el=args.el, er=args.er)
        if table:
            sol = {}
            for (q, s), (qo, so, do) in table.items():
                sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
            with open("solution_dfs.json", "w") as f:
                json.dump(sol, f, indent=2)
            print(f"\nSaved to solution_dfs.json")
            sys.exit(0)
