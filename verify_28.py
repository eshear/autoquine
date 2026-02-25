#!/usr/bin/env python3
"""Verify the (2,8) table against (6,2) on longer traces."""

import json
from collections import defaultdict
from tm_simulator import TM, TMRun, check_morita_reversibility
from algebraic_utm import NEARY_WOODS_62, make_62


def load_28():
    with open("solution_constrained_28.json") as f:
        sol = json.load(f)
    table = {}
    for key, val in sol.items():
        if key.startswith("_"):
            continue
        q, s = map(int, key.split(","))
        table[(q, s)] = (val["new_state"], val["new_symbol"], val["direction"])
    return table


def binary_to_octal(binary_tape):
    """Convert binary tape to octal (3 bits per symbol, LSB first)."""
    # Pad to multiple of 3
    padded = list(binary_tape)
    while len(padded) % 3 != 0:
        padded.append(0)
    octal = []
    for i in range(0, len(padded), 3):
        octal.append(padded[i] + 2 * padded[i+1] + 4 * padded[i+2])
    return octal


def octal_to_binary(octal_tape, n_blocks=None):
    """Convert octal tape to binary."""
    if n_blocks is None:
        n_blocks = len(octal_tape)
    binary = []
    for i in range(n_blocks):
        val = octal_tape[i] if i < len(octal_tape) else 0
        binary.extend([(val >> j) & 1 for j in range(3)])
    return binary


def run_comparison(binary_tape, head_pos, state_62, max_62_steps=200):
    """Run both (6,2) and (2,8) and compare at block-boundary crossings."""
    table_28 = load_28()
    tm62 = make_62()
    tm28 = TM(table_28, 2, 8, blank=0, name="(2,8)")

    # Pad tape
    padded = list(binary_tape)
    while len(padded) % 3 != 0:
        padded.append(0)
    n_blocks = len(padded) // 3

    octal_tape = binary_to_octal(padded)

    # Run (6,2)
    run62 = TMRun(tm62, tape=list(padded), head=head_pos, state=state_62)

    # For (2,8): start at the corresponding block
    block_start = head_pos // 3
    run28 = TMRun(tm28, tape=list(octal_tape), head=block_start, state=0)

    # Run (6,2) and track crossings
    prev_block = head_pos // 3
    crossing_count = 0
    match_count = 0
    mismatch_count = 0

    for step in range(1, max_62_steps + 1):
        if not run62.step():
            break
        curr_block = run62.head // 3
        if curr_block != prev_block:
            crossing_count += 1

            # Step the (2,8) machine
            if not run28.step():
                print(f"  (2,8) halted at crossing {crossing_count}")
                break

            # Compare tape content at this crossing point
            oct_from_62 = binary_to_octal([run62.tape[i] for i in range(len(padded))])
            oct_from_28 = [run28.tape[i] for i in range(n_blocks)]

            # Compare the block that was just modified
            modified_block = prev_block
            if oct_from_62[modified_block] == oct_from_28[modified_block]:
                match_count += 1
            else:
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(f"  MISMATCH at crossing {crossing_count} (step {step})")
                    print(f"    Modified block {modified_block}: "
                          f"(6,2)={oct_from_62[modified_block]} vs "
                          f"(2,8)={oct_from_28[modified_block]}")
                    # Show surrounding context
                    lo = max(0, modified_block - 2)
                    hi = min(n_blocks, modified_block + 3)
                    print(f"    (6,2) tape[{lo}:{hi}]: {oct_from_62[lo:hi]}")
                    print(f"    (2,8) tape[{lo}:{hi}]: {oct_from_28[lo:hi]}")

            prev_block = curr_block

    return crossing_count, match_count, mismatch_count


def main():
    table_28 = load_28()

    print("=" * 60)
    print("Verification: (2,8) table vs Neary-Woods (6,2)")
    print("=" * 60)

    # Print (2,8) table
    dn = {-1: 'L', 1: 'R'}
    print("\n(2,8) Transition table:")
    for q in range(2):
        for s in range(8):
            nq, ns, d = table_28[(q, s)]
            print(f"  q={q} sym={s}({s:03b}) → q={nq} sym={ns}({ns:03b}) {dn[d]}")
        print()

    # Morita check
    rev_ok, bad_group, bad_entries = check_morita_reversibility(table_28, 2, 8)
    print(f"Morita reversible: {'YES' if rev_ok else 'NO'}")

    # Group analysis
    groups = defaultdict(list)
    for (q, s), (nq, ns, d) in table_28.items():
        groups[(nq, d)].append((q, s, ns))
    print("\nMorita groups:")
    for (nq, d) in sorted(groups.keys()):
        entries = groups[(nq, d)]
        written = [e[2] for e in entries]
        unique = len(set(written))
        ok = "OK" if unique == len(written) else f"CONFLICT ({len(written)-unique} dupes)"
        print(f"  (nq={nq}, d={dn[d]}): {ok}")
        for e in entries:
            print(f"    ({e[0]},{e[1]}) writes {e[2]}")

    # Test 1: Paper's Rule 110 encoding
    print("\n--- Test 1: Paper's Rule 110 encoding (60 steps) ---")
    left_word = [0, 0, 0, 0, 0, 1, 0, 1]
    right_word = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]

    tape = []
    for _ in range(4):
        tape.extend(left_word)
    tape.extend([0, 0, 0, 0, 0, 0, 1, 1])
    head = len(tape) - 1
    for _ in range(4):
        tape.extend(right_word)

    crossings, matches, mismatches = run_comparison(tape, head, 0, max_62_steps=60)
    print(f"  Crossings: {crossings}, Matches: {matches}, Mismatches: {mismatches}")

    # Test 2: Longer run
    print("\n--- Test 2: Longer run (200 steps) ---")
    tape2 = []
    for _ in range(6):
        tape2.extend(left_word)
    tape2.extend([0, 0, 0, 0, 0, 0, 1, 1])
    head2 = len(tape2) - 1
    for _ in range(6):
        tape2.extend(right_word)

    crossings2, matches2, mismatches2 = run_comparison(tape2, head2, 0, max_62_steps=200)
    print(f"  Crossings: {crossings2}, Matches: {matches2}, Mismatches: {mismatches2}")

    # Test 3: Different initial content
    print("\n--- Test 3: All-zeros tape ---")
    tape3 = [0] * 60
    crossings3, matches3, mismatches3 = run_comparison(tape3, 30, 0, max_62_steps=100)
    print(f"  Crossings: {crossings3}, Matches: {matches3}, Mismatches: {mismatches3}")

    # Test 4: Direct (2,8) execution trace
    print("\n--- Test 4: Direct (2,8) execution trace ---")
    tm28 = TM(table_28, 2, 8, blank=0, name="(2,8)")
    oct_tape = binary_to_octal(tape)
    run = TMRun(tm28, tape=list(oct_tape), head=len(tape) // 3 - 1, state=0)
    run.run(max_steps=30, trace=True)

    if mismatches + mismatches2 + mismatches3 == 0:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print(f"\n*** {mismatches + mismatches2 + mismatches3} TOTAL MISMATCHES ***")


if __name__ == "__main__":
    main()
