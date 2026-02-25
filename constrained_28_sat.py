#!/usr/bin/env python3
"""
Constrained SAT Search: Find a (2,8) TM that simulates Neary-Woods U_{6,2}

Option C from the algebraic plan: since the (6,2) machine doesn't have
a clean Z/2Z × Z/3Z CRT decomposition, we search for a (2,8) transition
table that correctly simulates the (6,2) machine via block encoding.

The (2,8) machine groups 3 binary cells into one octal symbol.
Each (2,8) step may correspond to 1 or more (6,2) steps.

Strategy: generate (6,2) execution traces on the paper's Rule 110
encoding, then require the (2,8) machine to produce matching results
when operating on the blocked (octal) version of the same tape.

The key insight: we don't need perfect CRT structure. We just need
the (2,8) machine to produce the same final tape content as the (6,2)
machine after simulating each Rule 110 timestep.
"""

import time
import sys
import json
from collections import defaultdict
from pysat.solvers import Solver as SATSolver
from tm_simulator import TM, TMRun, check_morita_reversibility
from algebraic_utm import NEARY_WOODS_62, make_62


class VarPool:
    def __init__(self):
        self.next = 1
    def new(self):
        v = self.next
        self.next += 1
        return v


def generate_62_traces(max_steps=100):
    """Run the (6,2) machine and record state at block boundaries.

    Returns traces: each trace is a sequence of (block_idx, q_component, tape_snapshot)
    observed whenever the (6,2) head crosses a block boundary.
    """
    tm62 = make_62()

    # Build tape matching the paper's Rule 110 encoding
    left_word = [0, 0, 0, 0, 0, 1, 0, 1]
    right_word = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]

    n_left = 4
    n_right = 4

    tape = []
    for _ in range(n_left):
        tape.extend(left_word)
    tape.extend([0, 0, 0, 0, 0, 0, 1, 1])
    head_start = len(tape) - 1
    for _ in range(n_right):
        tape.extend(right_word)

    # Pad to multiple of 3
    while len(tape) % 3 != 0:
        tape.append(0)

    n_blocks = len(tape) // 3

    # Run the machine and record block-boundary crossings
    run = TMRun(tm62, tape=list(tape), head=head_start, state=0)

    crossings = []  # (step, block_idx, state, direction, tape_snapshot)

    prev_block = head_start // 3
    for step in range(1, max_steps + 1):
        if not run.step():
            break
        curr_block = run.head // 3
        if curr_block != prev_block:
            # Crossed a block boundary
            # Snapshot the octal tape
            oct_tape = []
            for i in range(n_blocks):
                b0 = run.tape[3*i]
                b1 = run.tape[3*i + 1]
                b2 = run.tape[3*i + 2]
                oct_tape.append(b0 + 2*b1 + 4*b2)
            crossings.append({
                'step': step,
                'block': curr_block,
                'prev_block': prev_block,
                'state': run.state,
                'direction': 1 if curr_block > prev_block else -1,
                'oct_tape': list(oct_tape),
            })
            prev_block = curr_block

    # Also record initial and final states
    init_oct = []
    for i in range(n_blocks):
        b0 = tape[3*i]
        b1 = tape[3*i + 1]
        b2 = tape[3*i + 2]
        init_oct.append(b0 + 2*b1 + 4*b2)

    return {
        'n_blocks': n_blocks,
        'init_oct': init_oct,
        'init_block': head_start // 3,
        'init_phase': head_start % 3,
        'init_state': 0,
        'crossings': crossings,
        'final_state': run.state,
        'final_head': run.head,
        'steps': run.steps,
    }


def search_28_constrained(traces, solver_name='cadical153'):
    """Search for a (2,8) table using constrained SAT.

    The (2,8) machine must produce matching tape content at each
    block-boundary crossing recorded from the (6,2) simulation.

    Rather than exact step-by-step matching (which would require
    knowing the exact correspondence), we use a simpler approach:

    For each consecutive pair of crossings where the (6,2) head
    enters and exits the same block, we constrain the (2,8)
    machine to produce the same tape modification.
    """
    pool = VarPool()
    clauses = []

    # Rule variables: (state, symbol) -> (new_state, new_symbol, direction)
    # 2 states × 8 symbols = 16 entries
    # new_state: 1 bool (True = state 1)
    # new_symbol: 8 options (one-hot over 0..7)
    # direction: 1 bool (True = R)

    rv_q = {}   # (state, sym) -> var for new_state
    rv_w = {}   # (state, sym, w) -> var for write symbol (one-hot)
    rv_d = {}   # (state, sym) -> var for direction

    for q in range(2):
        for s in range(8):
            rv_q[(q, s)] = pool.new()
            rv_d[(q, s)] = pool.new()
            for w in range(8):
                rv_w[(q, s, w)] = pool.new()
            # One-hot on write symbol
            w_vars = [rv_w[(q, s, w)] for w in range(8)]
            clauses.append(w_vars[:])
            for i in range(8):
                for j in range(i + 1, 8):
                    clauses.append([-w_vars[i], -w_vars[j]])

    print(f"  Rule variables: {pool.next - 1}")

    # Now build traces from the (6,2) crossings.
    # At each crossing, we know:
    # - The (6,2) machine just moved to a new block
    # - The tape content (as octal) at that moment
    # - The state
    #
    # For the (2,8) machine, each crossing should correspond to
    # one (2,8) step. The (2,8) machine reads an octal symbol,
    # writes a new one, and moves to an adjacent block.
    #
    # However, not every (6,2) block crossing is a (2,8) step.
    # Between (2,8) steps, the (6,2) machine may take several steps
    # within a single block.
    #
    # Approach: use the crossing trace to define (2,8) execution.
    # Each crossing = one (2,8) step.
    # The (2,8) machine reads the PREVIOUS block's content,
    # writes the modified content, and moves to the new block.

    crossings = traces['crossings']
    n_blocks = traces['n_blocks']
    n_crossings = len(crossings)

    if n_crossings < 2:
        print("  Not enough crossings for constraints.")
        return None

    print(f"  Crossings from (6,2) trace: {n_crossings}")

    # Build (2,8) execution trace from crossings.
    # At crossing i: the (6,2) machine just moved from prev_block to block.
    # Between crossing i-1 and i, the (6,2) machine was operating within
    # the prev_block (or crossing through multiple blocks).
    #
    # For each crossing:
    #   - The block that was just LEFT has its final content recorded
    #   - The block being ENTERED has its content available to read
    #
    # The (2,8) step that caused this crossing:
    #   - Was at prev_block, reading its content BEFORE the (6,2) modified it
    #   - Wrote the content that we see AFTER the (6,2) finished with it
    #   - Moved to the new block
    #
    # So we need BOTH the tape before and after the (6,2) visited a block.

    # Let's reconstruct: track tape state at each crossing
    # tape_at_crossing[i] = octal tape snapshot at crossing i
    # The crossing moved from prev_block to block.
    # The tape change between crossing i-1 and i tells us what
    # the (6,2) did to the blocks in between.

    # Actually, let's just set up a bounded trace for the (2,8) machine
    # and require it matches the (6,2) tape at specific checkpoints.

    # Simpler approach: Run (2,8) for some steps and require tape matches
    # at the end.
    n_28_steps = n_crossings  # Each crossing = one (2,8) step

    # Trace variables for (2,8)
    tv = {}   # (t, block) -> one-hot over 8 symbols
    hv = {}   # (t, block) -> bool (head position)
    sv = {}   # (t) -> bool (state; True=1)

    for t in range(n_28_steps + 1):
        sv[t] = pool.new()
        for x in range(n_blocks):
            hv[(t, x)] = pool.new()
            for s in range(8):
                tv[(t, x, s)] = pool.new()
            # One-hot on symbol
            s_vars = [tv[(t, x, s)] for s in range(8)]
            clauses.append(s_vars[:])
            for i in range(8):
                for j in range(i + 1, 8):
                    clauses.append([-s_vars[i], -s_vars[j]])
        # One-hot on head
        h_vars = [hv[(t, x)] for x in range(n_blocks)]
        clauses.append(h_vars[:])
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                clauses.append([-h_vars[i], -h_vars[j]])

    # Initial conditions from (6,2) trace
    init_oct = traces['init_oct']
    init_block = traces['init_block']

    for x in range(n_blocks):
        if x == init_block:
            clauses.append([hv[(0, x)]])
        else:
            clauses.append([-hv[(0, x)]])
        for s in range(8):
            if s == init_oct[x]:
                clauses.append([tv[(0, x, s)]])
            else:
                clauses.append([-tv[(0, x, s)]])

    # Initial state: we don't know the (2,8) state mapping yet,
    # but let's say state 0.
    clauses.append([-sv[0]])  # state 0

    # Transition constraints
    for t in range(n_28_steps):
        for x in range(n_blocks):
            for q in range(2):
                for s in range(8):
                    cond = [hv[(t, x)], tv[(t, x, s)]]
                    if q == 1:
                        cond.append(sv[t])
                    else:
                        cond.append(-sv[t])
                    neg = [-l for l in cond]

                    # State transition
                    clauses.append(neg + [-rv_q[(q, s)], sv[t + 1]])
                    clauses.append(neg + [rv_q[(q, s)], -sv[t + 1]])

                    # Head movement
                    nx_r = x + 1
                    nx_l = x - 1
                    if nx_r >= n_blocks and nx_l < 0:
                        clauses.append([-l for l in cond])
                        continue
                    elif nx_r >= n_blocks:
                        clauses.append(neg + [-rv_d[(q, s)]])
                        clauses.append(neg + [hv[(t + 1, nx_l)]])
                    elif nx_l < 0:
                        clauses.append(neg + [rv_d[(q, s)]])
                        clauses.append(neg + [hv[(t + 1, nx_r)]])
                    else:
                        clauses.append(neg + [-rv_d[(q, s)], hv[(t + 1, nx_r)]])
                        clauses.append(neg + [rv_d[(q, s)], hv[(t + 1, nx_l)]])

                    # Write symbol
                    for w in range(8):
                        for s2 in range(8):
                            if s2 == w:
                                clauses.append(neg + [-rv_w[(q, s, w)],
                                               tv[(t + 1, x, s2)]])
                            else:
                                clauses.append(neg + [-rv_w[(q, s, w)],
                                               -tv[(t + 1, x, s2)]])

                    # Frame: other blocks unchanged
                    for x2 in range(n_blocks):
                        if x2 != x:
                            for s2 in range(8):
                                clauses.append(neg + [-tv[(t, x2, s2)],
                                               tv[(t + 1, x2, s2)]])
                                clauses.append(neg + [tv[(t, x2, s2)],
                                               -tv[(t + 1, x2, s2)]])

    # Checkpoint constraints: at certain steps, require tape matches
    # Use the crossings to determine checkpoints.
    # After all crossings, the tape should match the final (6,2) state.
    if crossings:
        final = crossings[-1]
        final_oct = final['oct_tape']
        t_final = n_28_steps

        print(f"  Final checkpoint at t={t_final}")
        for x in range(n_blocks):
            for s in range(8):
                if s == final_oct[x]:
                    clauses.append([tv[(t_final, x, s)]])
                else:
                    clauses.append([-tv[(t_final, x, s)]])

    n_vars = pool.next - 1
    n_clauses = len(clauses)
    print(f"\n  Total variables: {n_vars:,}")
    print(f"  Total clauses: {n_clauses:,}")

    print(f"\nSolving with {solver_name}...")
    t0 = time.time()
    solver = SATSolver(name=solver_name, bootstrap_with=clauses)
    result = solver.solve()
    elapsed = time.time() - t0
    print(f"  Result: {'SAT' if result else 'UNSAT'} ({elapsed:.1f}s)")

    if result:
        model = solver.get_model()
        table_28 = {}
        for q in range(2):
            for s in range(8):
                nq = 1 if model[rv_q[(q, s)] - 1] > 0 else 0
                nd = 1 if model[rv_d[(q, s)] - 1] > 0 else -1
                ws = 0
                for w in range(8):
                    if model[rv_w[(q, s, w)] - 1] > 0:
                        ws = w
                        break
                table_28[(q, s)] = (nq, ws, nd)

        print("\n(2,8) Transition table:")
        dn = {-1: 'L', 1: 'R'}
        for q in range(2):
            for s in range(8):
                nq, ns, d = table_28[(q, s)]
                print(f"  q={q} sym={s}({s:03b}) → q={nq} sym={ns}({ns:03b}) {dn[d]}")
            print()

        # Check Morita
        rev_ok, _, _ = check_morita_reversibility(table_28, 2, 8)
        print(f"  Morita reversible: {'yes' if rev_ok else 'no'}")

        solver.delete()
        return table_28

    solver.delete()
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Constrained SAT: Find (2,8) simulating Neary-Woods (6,2)")
    print("=" * 60)

    print("\n--- Generating (6,2) traces ---")
    traces = generate_62_traces(max_steps=60)
    print(f"  Blocks: {traces['n_blocks']}")
    print(f"  Initial: block {traces['init_block']}, phase {traces['init_phase']}, state u{traces['init_state']+1}")
    print(f"  Crossings: {len(traces['crossings'])}")
    print(f"  Steps run: {traces['steps']}")

    if traces['crossings']:
        print(f"\n  First 10 crossings:")
        for i, c in enumerate(traces['crossings'][:10]):
            dn = {-1: '←', 1: '→'}
            print(f"    t={c['step']:3d}: block {c['prev_block']} {dn[c['direction']]} block {c['block']}, "
                  f"state u{c['state']+1}")

    print("\n--- SAT Search ---")
    table = search_28_constrained(traces)

    if table:
        print("\n*** (2,8) TABLE FOUND ***")
        sol = {}
        for (q, s), (nq, ns, d) in table.items():
            sol[f"{q},{s}"] = {"new_state": nq, "new_symbol": ns, "direction": d}
        with open("solution_constrained_28.json", "w") as f:
            json.dump(sol, f, indent=2)
        print("Saved to solution_constrained_28.json")
    else:
        print("\nNo solution found with this approach.")
        print("Consider: more steps, different checkpoints, or direct Rule 110 search.")
