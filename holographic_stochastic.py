#!/usr/bin/env python3
"""
(2, 8) Holographic TM — Stochastic Search

Instead of Z3 (which times out on 6 simultaneous scenarios),
use simulated annealing / hill climbing to find a transition table.

Fitness = number of scenarios correctly completed.
"""

import json
import random
import math
import time
import sys


def random_table(max_sing=4):
    """Generate a random transition table respecting constraints."""
    table = {}
    # Fixed boundary rules
    table[(0, 0)] = (0, 0, 1)       # SR+0: pass through
    table[(1, 7)] = (1, 7, -1)      # SL+7: pass through
    table[(0, 7)] = (1, random.randint(1, 6), -1)   # SR+7: bounce
    table[(1, 0)] = (0, random.randint(1, 6), 1)     # SL+0: bounce

    # Payload rules (symbols 1-6)
    for q in range(2):
        for s in range(1, 7):
            table[(q, s)] = (
                random.randint(0, 1),
                random.randint(1, 6),
                random.choice([-1, 1]),
            )
    return table


def mutate_table(table, n_mutations=1):
    """Mutate a transition table."""
    table = dict(table)
    for _ in range(n_mutations):
        # Pick a random mutable entry
        while True:
            q = random.randint(0, 1)
            s = random.randint(0, 7)
            if (q, s) == (0, 0) or (q, s) == (1, 7):
                continue  # Can't mutate fixed pass-through rules
            break

        qo, so, do = table[(q, s)]

        # Choose what to mutate
        what = random.randint(0, 2)
        if (q, s) == (0, 7):
            # SR+7: can only change write symbol
            table[(q, s)] = (1, random.randint(1, 6), -1)
        elif (q, s) == (1, 0):
            # SL+0: can only change write symbol
            table[(q, s)] = (0, random.randint(1, 6), 1)
        else:
            if what == 0:
                table[(q, s)] = (1 - qo, so, do)  # Flip state
            elif what == 1:
                table[(q, s)] = (qo, random.randint(1, 6), do)  # Change write
            else:
                table[(q, s)] = (qo, so, -do)  # Flip direction
    return table


def sim_fast(table, tape, head, state, max_steps, x_max):
    """Fast simulation returning final state."""
    for _ in range(max_steps):
        if head < 0 or head >= x_max:
            return tape, head, state, False  # Out of bounds
        sym = tape[head]
        nq, ns, nd = table[(state, sym)]
        tape[head] = ns
        state = nq
        head += nd
    return tape, head, state, True


def evaluate_scenario(table, pay_in, pay_out, max_steps, x_max, left_pad):
    """Evaluate how well the table handles one scenario.
    Returns a score: 1.0 = perfect, 0.0 = complete failure."""
    pay_len = len(pay_in)
    tape = [0]*left_pad + list(pay_in) + [7]*(x_max - left_pad - pay_len)
    tape, head, state, completed = sim_fast(
        table, tape, left_pad, 0, max_steps, x_max
    )

    if not completed:
        return 0.0

    # Check payload match
    actual = tape[left_pad:left_pad + len(pay_out)]
    payload_match = sum(1 for a, b in zip(actual, pay_out) if a == b) / len(pay_out)

    # Check state and head
    state_ok = 1.0 if state == 0 else 0.0
    head_ok = 1.0 if head == left_pad else max(0.0, 1.0 - abs(head - left_pad) * 0.1)

    # Check boundary cleanup
    left_ok = sum(1 for x in range(left_pad) if tape[x] == 0) / max(left_pad, 1)
    right_ok = sum(1 for x in range(left_pad + pay_len, x_max) if tape[x] == 7)
    right_ok /= max(x_max - left_pad - pay_len, 1)

    # Weighted score
    return (payload_match * 0.5 + state_ok * 0.15 + head_ok * 0.15 +
            left_ok * 0.1 + right_ok * 0.1)


def evaluate_table(table, scenarios, max_steps, x_max, left_pad):
    """Evaluate table on all scenarios."""
    total = 0.0
    perfect = 0
    for pay_in, pay_out, desc in scenarios:
        score = evaluate_scenario(table, pay_in, pay_out, max_steps, x_max, left_pad)
        total += score
        if score > 0.999:
            perfect += 1
    return total, perfect


def verify_table(table, scenarios, max_steps, x_max, left_pad):
    """Strict verification."""
    all_ok = True
    for pay_in, pay_out, desc in scenarios:
        pay_len = len(pay_in)
        tape = [0]*left_pad + list(pay_in) + [7]*(x_max - left_pad - pay_len)
        tape, head, state, completed = sim_fast(
            table, list(tape), left_pad, 0, max_steps, x_max
        )
        actual = tape[left_pad:left_pad + len(pay_out)]
        ok = (actual == list(pay_out) and head == left_pad and state == 0 and
              all(tape[x] == 0 for x in range(left_pad)) and
              all(tape[x] == 7 for x in range(left_pad + pay_len, x_max)))
        if not ok:
            all_ok = False
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}: {list(pay_in)} -> {actual} "
              f"h={head} s={state}")
    return all_ok


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


def search_sa(scenarios, max_steps=20, x_max=14, left_pad=4,
              max_iter=10000000, temp_start=2.0, temp_end=0.01,
              report_every=100000):
    """Simulated annealing search."""
    print(f"Stochastic search: {len(scenarios)} scenarios, "
          f"max_steps={max_steps}, tape={x_max}, pad={left_pad}")
    print(f"Max iterations: {max_iter}")

    best_table = random_table()
    best_score, best_perfect = evaluate_table(
        best_table, scenarios, max_steps, x_max, left_pad)

    current_table = dict(best_table)
    current_score = best_score

    t0 = time.time()
    accepted = 0

    for iteration in range(max_iter):
        # Temperature schedule
        progress = iteration / max_iter
        temp = temp_start * (temp_end / temp_start) ** progress

        # Mutate
        n_mut = 1 if random.random() < 0.7 else random.randint(2, 3)
        candidate = mutate_table(current_table, n_mutations=n_mut)
        cand_score, cand_perfect = evaluate_table(
            candidate, scenarios, max_steps, x_max, left_pad)

        # Accept?
        delta = cand_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            current_table = candidate
            current_score = cand_score
            accepted += 1

            if cand_score > best_score or cand_perfect > best_perfect:
                best_table = dict(candidate)
                best_score = cand_score
                best_perfect = cand_perfect
                elapsed = time.time() - t0
                print(f"  iter={iteration:8d} score={best_score:.4f} "
                      f"perfect={best_perfect}/{len(scenarios)} "
                      f"temp={temp:.4f} ({elapsed:.1f}s)")

                if best_perfect == len(scenarios):
                    print(f"\n*** ALL SCENARIOS PERFECT ***")
                    return best_table

        if (iteration + 1) % report_every == 0:
            elapsed = time.time() - t0
            rate = (iteration + 1) / elapsed
            print(f"  iter={iteration+1:8d} best={best_score:.4f} "
                  f"perfect={best_perfect}/{len(scenarios)} "
                  f"cur={current_score:.4f} temp={temp:.4f} "
                  f"accept={accepted/(iteration+1):.3f} ({rate:.0f}/s)")

    return best_table


def test_generalization(table, max_steps):
    print("\n--- Generalization test ---")
    for a_val in [1, 2, 3, 4, 5, 8]:
        x_big = 80
        lp = 25
        payload = [1]*a_val + [3]
        tape = [0]*lp + list(payload) + [7]*(x_big - lp - len(payload))
        total_steps = max_steps * (2*a_val + 2) * 5
        tape, head, state, _ = sim_fast(
            table, tape, lp, 0, total_steps, x_big)

        # Check if cycle completed
        l = 0
        while l < x_big and tape[l] == 0: l += 1
        r = x_big - 1
        while r >= 0 and tape[r] == 7: r -= 1
        pay = tape[l:r+1]
        ok = (pay == [1]*a_val + [3] and head == lp and state == 0)
        # Also check at intermediate steps
        tape2 = [0]*lp + list(payload) + [7]*(x_big - lp - len(payload))
        cycle_found = False
        h2, s2 = lp, 0
        for t in range(1, total_steps + 1):
            if h2 < 0 or h2 >= x_big: break
            nq, ns, nd = table[(s2, tape2[h2])]
            tape2[h2] = ns; s2 = nq; h2 += nd
            if h2 == lp and s2 == 0:
                pay2 = tape2[lp:lp + a_val + 1]
                if pay2 == [1]*a_val + [3]:
                    l2 = 0
                    while l2 < x_big and tape2[l2] == 0: l2 += 1
                    r2 = x_big - 1
                    while r2 >= 0 and tape2[r2] == 7: r2 -= 1
                    if l2 == lp and r2 == lp + a_val:
                        print(f"  A={a_val}: FULL CYCLE at t={t}")
                        cycle_found = True
                        break

        if not cycle_found:
            print(f"  A={a_val}: NO CYCLE after {total_steps} steps. pay={pay[:15]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--encoding", choices=["sliding", "old"], default="sliding")
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--x-max", type=int, default=14)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=50000000)
    p.add_argument("--temp-start", type=float, default=2.0)
    p.add_argument("--temp-end", type=float, default=0.005)
    args = p.parse_args()

    if args.encoding == "sliding":
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

    print(f"Encoding: {args.encoding}")
    print(f"Scenarios:")
    for pi, po, desc in scenarios:
        print(f"  {pi} -> {po}  ({desc})")
    print()

    table = search_sa(
        scenarios,
        max_steps=args.max_steps,
        x_max=args.x_max,
        left_pad=args.pad,
        max_iter=args.max_iter,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
    )

    print_table(table)
    print("\n--- Final verification ---")
    ok = verify_table(table, scenarios, args.max_steps, args.x_max, args.pad)

    if ok:
        print("\n*** ALL SCENARIOS VERIFIED ***")
        test_generalization(table, args.max_steps)

        sol = {}
        for (q, s), (qo, so, do) in table.items():
            sol[f"{q},{s}"] = {"new_state": qo, "new_symbol": so, "direction": do}
        with open("solution_stochastic.json", "w") as f:
            json.dump(sol, f, indent=2)
        print("\nSaved to solution_stochastic.json")
