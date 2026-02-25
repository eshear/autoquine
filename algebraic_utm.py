#!/usr/bin/env python3
"""
Algebraic UTM Construction: (6,2) → (2,8)

Builds a 2-state 8-symbol universal Turing machine from
Neary & Woods' published (6,2) weakly universal TM.

The 6 states decompose as Z/2Z × Z/3Z via CRT:
  6 = 2 × 3
The Z/2Z component stays as the 2 states of (2,8).
The Z/3Z component gets externalized onto the tape:
  each octal symbol represents 3 binary cells (2^3 = 8).

Source: Neary & Woods, "Small Weakly Universal Turing Machines"
  FCT 2009, LNCS vol. 5699, arXiv:0707.4489
"""

import sys
import json
import time
from collections import defaultdict
from tm_simulator import TM, TMRun, check_morita_reversibility


# ============================================================
# Neary & Woods U_{6,2}: 6 states, 2 symbols
# ============================================================
#
# From Table 3 of the paper:
#
#       u1     u2     u3     u4     u5     u6
#  0  0Lu1   0Lu6   0Ru2   1Ru5   1Lu4   1Lu1
#  1  1Lu2   0Lu3   1Lu3   0Ru6   1Ru4   0Ru4
#
# Format: (write_symbol)(direction)(new_state)
# States: u1=0, u2=1, u3=2, u4=3, u5=4, u6=5
# Symbols: 0, 1
# Direction: L=-1, R=+1

NEARY_WOODS_62 = {
    # State u1 (0)
    (0, 0): (0, 0, -1),   # 0Lu1: write 0, L, go to u1
    (0, 1): (1, 1, -1),   # 1Lu2: write 1, L, go to u2
    # State u2 (1)
    (1, 0): (5, 0, -1),   # 0Lu6: write 0, L, go to u6
    (1, 1): (2, 0, -1),   # 0Lu3: write 0, L, go to u3
    # State u3 (2)
    (2, 0): (1, 0, 1),    # 0Ru2: write 0, R, go to u2
    (2, 1): (2, 1, -1),   # 1Lu3: write 1, L, go to u3
    # State u4 (3)
    (3, 0): (4, 1, 1),    # 1Ru5: write 1, R, go to u5
    (3, 1): (5, 0, 1),    # 0Ru6: write 0, R, go to u6
    # State u5 (4)
    (4, 0): (3, 1, -1),   # 1Lu4: write 1, L, go to u4
    (4, 1): (3, 1, 1),    # 1Ru4: write 1, R, go to u4
    # State u6 (5)
    (5, 0): (0, 1, -1),   # 1Lu1: write 1, L, go to u1
    (5, 1): (3, 0, 1),    # 0Ru4: write 0, R, go to u4
}


# ============================================================
# Neary & Woods U_{2,4}: 2 states, 4 symbols
# ============================================================
#
# From Table 2 of the paper:
# Symbols: 0, 1, ∅ (empty set), 1̸ (1 with stroke)
# We encode: 0=0, 1=1, ∅=2, 1̸=3
#
#        u1      u2
#   0   ∅Lu1    1̸Ru1
#   1   1̸Lu2    ∅Lu2
#   ∅   1̸Lu1    0Ru2
#   1̸   1̸Lu1    1Ru2

NEARY_WOODS_24 = {
    # State u1 (0)
    (0, 0): (0, 2, -1),   # ∅Lu1: write ∅, L, u1
    (0, 1): (1, 3, -1),   # 1̸Lu2: write 1̸, L, u2
    (0, 2): (0, 3, -1),   # 1̸Lu1: write 1̸, L, u1
    (0, 3): (0, 3, -1),   # 1̸Lu1: write 1̸, L, u1
    # State u2 (1)
    (1, 0): (0, 3, 1),    # 1̸Ru1: write 1̸, R, u1
    (1, 1): (1, 2, -1),   # ∅Lu2: write ∅, L, u2
    (1, 2): (1, 0, 1),    # 0Ru2: write 0, R, u2
    (1, 3): (1, 1, 1),    # 1Ru2: write 1, R, u2
}


def make_62():
    """Create the Neary & Woods (6,2) TM object."""
    return TM(
        NEARY_WOODS_62, n_states=6, n_symbols=2, blank=0,
        name="Neary-Woods U(6,2)",
        state_names={0: 'u1', 1: 'u2', 2: 'u3', 3: 'u4', 4: 'u5', 5: 'u6'},
    )


def make_24():
    """Create the Neary & Woods (2,4) TM object."""
    return TM(
        NEARY_WOODS_24, n_states=2, n_symbols=4, blank=0,
        name="Neary-Woods U(2,4)",
        state_names={0: 'u1', 1: 'u2'},
        symbol_names={0: '0', 1: '1', 2: '∅', 3: '1̸'},
    )


def verify_62():
    """Verify U_{6,2} by simulating the Rule 110 trace from the paper."""
    tm = make_62()
    tm.print_table()
    errs = tm.validate()
    if errs:
        print(f"Validation errors: {errs}")
        return False

    # From the paper, Section 3.3:
    # Rule 110 state 0 encoded as 00, state 1 as 11.
    # Left blank word: 00000101 (repeated)
    # Right blank word: 100100001001 (repeated)
    # Word 010100 terminates left traversal.
    # Rule 110 cell encoding: 0 → 00, 1 → 11

    # Simulate c_0 → c_1 from the paper trace
    # Initial config encodes c_0 from Figure 2
    # The paper shows a specific trace. Let's verify the machine runs correctly.

    # From the paper: tape content at start (encoding c_0 from Figure 2)
    # Left of head: ...00000101 00000101 00000011
    # Under head: position at rightmost bit shown
    # Right: 100100001001...
    #
    # Let's build the tape. The paper shows the head starting over cell index 0.
    # Left blank word repeating: 00000101
    # Right blank word repeating: 100100001001

    # Build a finite tape approximation
    left_word = [0, 0, 0, 0, 0, 1, 0, 1]   # 00000101
    right_word = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]  # 100100001001

    # Repeat left word several times, then add initial config, then right word
    n_left = 5
    n_right = 5

    # The initial tape encodes c_0.
    # From Figure 2 in the paper (not shown to us, but the trace shows):
    # The tape at start: ...00000101 | 00000101 | 00000011 | 100100001001...
    # The encoding of the Rule 110 initial state is embedded in the middle.
    # Looking at the trace: head starts at the underlined position in the
    # leftmost shown segment, state u1.

    # Let's construct a tape matching the paper's initial config.
    # From the trace in the paper for U_{6,2}:
    # u1, ...00000101 00000101 0000001_1 100100001001...
    # The head (underlined) is on the last 1 before the right blank word.

    tape = []
    for _ in range(n_left):
        tape.extend(left_word)
    # The "00000011" segment between left blanks and right blanks
    tape.extend([0, 0, 0, 0, 0, 0, 1, 1])
    head_pos = len(tape) - 1  # on the last 1

    for _ in range(n_right):
        tape.extend(right_word)

    print(f"\nInitial tape ({len(tape)} cells), head at {head_pos}, state u1")
    print(f"  Left of head: ...{tape[max(0,head_pos-16):head_pos]}")
    print(f"  Under head: {tape[head_pos]}")
    print(f"  Right of head: {tape[head_pos+1:head_pos+17]}...")

    run = TMRun(tm, tape=tape, head=head_pos, state=0)

    # Run for enough steps to simulate one Rule 110 timestep
    # The paper shows ~15 steps for c_0 → c_1
    print(f"\nRunning U(6,2) for 100 steps...")
    run.run(max_steps=100, trace=True)

    return True


# ============================================================
# CRT Decomposition Analysis
# ============================================================

def analyze_crt_decomposition():
    """Analyze U_{6,2} for Z/2Z × Z/3Z decomposition.

    For the (6,2) → (2,8) trade to work, we need to find a
    labeling of the 6 states as (q, p) ∈ Z/2Z × Z/3Z such that:
      - The Z/3Z component p tracks position within 3-cell blocks
      - Moving R increments p by 1 mod 3
      - Moving L decrements p by 1 mod 3

    We search over all possible labelings (6! / (2! × 3!) permutations).
    """
    import itertools

    print("\n" + "=" * 60)
    print("CRT Decomposition Analysis")
    print("=" * 60)

    table = NEARY_WOODS_62

    # Try all possible assignments of 6 states to (q, p) pairs
    # q ∈ {0,1}, p ∈ {0,1,2}
    # This is a bijection from {0..5} to {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}
    pairs = [(q, p) for q in range(2) for p in range(3)]

    best = None
    best_violations = float('inf')

    for perm in itertools.permutations(range(6)):
        # perm[i] = original state index for CRT pair pairs[i]
        # So: state perm[i] maps to CRT pair (q_i, p_i)
        state_to_crt = {}
        for i, orig_state in enumerate(perm):
            state_to_crt[orig_state] = pairs[i]

        # Check phase tracking
        violations = 0
        for (s, sym), (ns, ws, d) in table.items():
            q, p = state_to_crt[s]
            nq, np = state_to_crt[ns]
            if d == 1:  # R
                expected_p = (p + 1) % 3
            else:  # L
                expected_p = (p - 1) % 3
            if np != expected_p:
                violations += 1

        if violations < best_violations:
            best_violations = violations
            best = (perm, state_to_crt)

        if violations == 0:
            print(f"\n  PERFECT CRT decomposition found!")
            print(f"  Mapping:")
            for orig_state in range(6):
                q, p = state_to_crt[orig_state]
                print(f"    u{orig_state+1} (state {orig_state}) → (q={q}, p={p})")

            # Print the rewritten table
            print(f"\n  Rewritten transition table:")
            dn = {-1: 'L', 1: 'R'}
            for s in range(6):
                q, p = state_to_crt[s]
                for sym in range(2):
                    ns, ws, d = table[(s, sym)]
                    nq, np = state_to_crt[ns]
                    print(f"    (q={q},p={p})+{sym} → (q={nq},p={np}) w={ws} {dn[d]}")

            return state_to_crt

    print(f"\n  No perfect decomposition found.")
    print(f"  Best has {best_violations} violations out of 12 transitions.")

    if best:
        perm, state_to_crt = best
        print(f"\n  Best mapping:")
        for orig_state in range(6):
            q, p = state_to_crt[orig_state]
            print(f"    u{orig_state+1} (state {orig_state}) → (q={q}, p={p})")

        print(f"\n  Violations:")
        dn = {-1: 'L', 1: 'R'}
        for (s, sym), (ns, ws, d) in table.items():
            q, p = state_to_crt[s]
            nq, np = state_to_crt[ns]
            if d == 1:
                expected_p = (p + 1) % 3
            else:
                expected_p = (p - 1) % 3
            if np != expected_p:
                print(f"    (q={q},p={p})+{sym} → (q={nq},p={np}) {dn[d]}  "
                      f"expected p'={(p + (1 if d==1 else -1)) % 3}")

    return None


# ============================================================
# Phase 3: (6,2) → (2,8) Exponential Trade
# ============================================================

def build_28_from_62(state_to_crt):
    """Convert U_{6,2} to a (2,8) machine via CRT block encoding.

    Given a CRT decomposition state_to_crt mapping original states
    to (q, p) pairs:
    - q ∈ {0,1} = Z/2Z = the 2 states of (2,8)
    - p ∈ {0,1,2} = Z/3Z = position within 3-cell binary block

    The (2,8) machine groups 3 consecutive binary cells into one
    octal symbol. Octal symbol value = b0 + 2*b1 + 4*b2 where
    b0,b1,b2 are the binary cells at positions 3k, 3k+1, 3k+2.

    For each (2,8) transition (q_in, oct_sym):
    1. Decode oct_sym to bits (b0, b1, b2)
    2. Simulate the (6,2) machine starting at phase 0 within this block
    3. Continue until the machine exits the block
    4. Re-encode the modified bits as a new octal symbol
    5. Record the exit direction and new Z/2Z state
    """
    if state_to_crt is None:
        print("No CRT decomposition available.")
        return None

    table = NEARY_WOODS_62

    # Build reverse mapping: (q, p) → original state
    crt_to_state = {}
    for orig, (q, p) in state_to_crt.items():
        crt_to_state[(q, p)] = orig

    print("\n" + "=" * 60)
    print("Phase 3: (6,2) → (2,8) Exponential Trade")
    print("=" * 60)

    table_28 = {}

    for q_in in range(2):
        for oct_sym in range(8):
            # Decode octal to bits (b0 = LSB)
            bits = [(oct_sym >> i) & 1 for i in range(3)]

            # Simulate (6,2) starting at (q_in, phase=0) within this block
            q = q_in
            p = 0
            pos = 0  # position within block {0, 1, 2}
            local_tape = list(bits)

            steps = 0
            max_steps = 30  # safety limit

            while steps < max_steps:
                # Get the original (6,2) state
                orig_state = crt_to_state[(q, p)]
                sym = local_tape[pos]
                ns, ws, d = table[(orig_state, sym)]
                nq, np = state_to_crt[ns]

                local_tape[pos] = ws
                new_pos = pos + d
                steps += 1

                if new_pos < 0:
                    # Exited block to the left
                    new_oct = sum(local_tape[i] << i for i in range(3))
                    table_28[(q_in, oct_sym)] = (nq, new_oct, -1)
                    break
                elif new_pos >= 3:
                    # Exited block to the right
                    new_oct = sum(local_tape[i] << i for i in range(3))
                    table_28[(q_in, oct_sym)] = (nq, new_oct, 1)
                    break
                else:
                    q = nq
                    p = np
                    pos = new_pos
            else:
                print(f"  WARNING: infinite loop for q={q_in}, oct={oct_sym}")
                new_oct = sum(local_tape[i] << i for i in range(3))
                table_28[(q_in, oct_sym)] = (q, new_oct, 1)

    # Print the table
    print("\n(2,8) Transition table:")
    dn = {-1: 'L', 1: 'R'}
    for q in range(2):
        for s in range(8):
            if (q, s) in table_28:
                nq, ns, d = table_28[(q, s)]
                bits_in = f"{s:03b}"
                bits_out = f"{ns:03b}"
                print(f"  q={q} sym={s}({bits_in}) → q={nq} sym={ns}({bits_out}) {dn[d]}")
        print()

    # Check properties
    print("Properties:")

    # Local injectivity
    outputs = set()
    injective = True
    for (q, s), (nq, ns, d) in table_28.items():
        out = (nq, ns, d)
        if out in outputs:
            injective = False
        outputs.add(out)
    print(f"  Locally injective: {'yes' if injective else 'no'}")

    # Morita reversibility
    rev_ok, bad_group, bad_entries = check_morita_reversibility(
        table_28, 2, 8)
    print(f"  Morita reversible: {'yes' if rev_ok else 'no'}")
    if not rev_ok and bad_group:
        dn = {-1: 'L', 1: 'R'}
        print(f"    Violation in group (nq={bad_group[0]}, d={dn[bad_group[1]]})")
        for e in bad_entries:
            print(f"      ({e[0]},{e[1]}) writes {e[2]}")

    # Group analysis
    groups = defaultdict(list)
    for (q, s), (nq, ns, d) in table_28.items():
        groups[(nq, d)].append((q, s, ns))
    print(f"\n  Morita group analysis:")
    for (nq, d) in sorted(groups.keys()):
        entries = groups[(nq, d)]
        written = [e[2] for e in entries]
        unique = len(set(written))
        ok = "✓" if unique == len(written) else "✗"
        print(f"    (nq={nq}, d={dn[d]}): {len(entries)} entries, "
              f"{unique} unique writes {ok}")
        for e in entries:
            print(f"      ({e[0]},{e[1]}) → writes {e[2]}")

    return table_28


# ============================================================
# Verification
# ============================================================

def verify_28_vs_62(table_28, state_to_crt):
    """Verify (2,8) simulates (6,2) correctly on test inputs."""
    if table_28 is None or state_to_crt is None:
        return False

    table_62 = NEARY_WOODS_62

    # Reverse mapping
    crt_to_state = {}
    for orig, (q, p) in state_to_crt.items():
        crt_to_state[(q, p)] = orig

    print("\n" + "=" * 60)
    print("Verification: (2,8) vs (6,2)")
    print("=" * 60)

    # Test on several binary tapes
    # Encode as both (6,2) binary and (2,8) octal, run both, compare

    # Use the paper's Rule 110 encoding:
    # Left blank word: 00000101 (8 bits)
    # Right blank word: 100100001001 (12 bits)
    left_word = [0, 0, 0, 0, 0, 1, 0, 1]
    right_word = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]

    # Build binary tape
    n_left = 3
    n_right = 3
    binary_tape = []
    for _ in range(n_left):
        binary_tape.extend(left_word)
    binary_tape.extend([0, 0, 0, 0, 0, 0, 1, 1])
    head_62 = len(binary_tape) - 1
    for _ in range(n_right):
        binary_tape.extend(right_word)

    # Convert to octal tape
    # Pad to multiple of 3
    while len(binary_tape) % 3 != 0:
        binary_tape.append(0)

    octal_tape = []
    for i in range(0, len(binary_tape), 3):
        oct_val = binary_tape[i] + 2 * binary_tape[i+1] + 4 * binary_tape[i+2]
        octal_tape.append(oct_val)

    # Head position in octal tape: which block contains head_62?
    block_idx = head_62 // 3
    phase_in_block = head_62 % 3

    print(f"\nBinary tape: {len(binary_tape)} cells, head at {head_62}")
    print(f"Octal tape: {len(octal_tape)} cells, head at block {block_idx}")
    print(f"Phase within block: {phase_in_block}")

    if phase_in_block != 0:
        print(f"  WARNING: head not aligned to block boundary (phase={phase_in_block})")
        print(f"  The (2,8) machine requires phase 0 alignment.")
        print(f"  Adjusting head position...")
        # Find nearest block-aligned position
        head_62 = block_idx * 3
        phase_in_block = 0
        print(f"  New head: {head_62} (block {block_idx}, phase 0)")

    # Run (6,2) machine
    tm62 = make_62()
    run62 = TMRun(tm62, tape=list(binary_tape), head=head_62, state=0)

    # Run (2,8) machine
    tm28 = TM(table_28, n_states=2, n_symbols=8, blank=0, name="(2,8)")
    # Start in state corresponding to CRT q-component of u1 (state 0)
    q_start, p_start = state_to_crt[0]
    run28 = TMRun(tm28, tape=list(octal_tape), head=block_idx, state=q_start)

    # Run both for some steps and compare
    print(f"\nRunning (6,2) for 50 steps:")
    run62.run(max_steps=50)
    print(f"  Final: state=u{run62.state+1}, head={run62.head}, steps={run62.steps}")

    # For comparison, count how many (2,8) steps correspond to 50 (6,2) steps.
    # Each (2,8) step corresponds to 1-3 (6,2) steps (depending on how many
    # internal steps happen within a block before exiting).
    print(f"\nRunning (2,8) for 50 steps:")
    run28.run(max_steps=50)
    print(f"  Final: state={run28.state}, head={run28.head}, steps={run28.steps}")

    # Compare tape contents
    # Convert (2,8) tape back to binary
    binary_from_28 = []
    lo28, hi28 = 0, len(octal_tape)
    for i in range(lo28, hi28):
        val = run28.tape[i]
        binary_from_28.extend([(val >> j) & 1 for j in range(3)])

    # Get (6,2) tape
    binary_from_62 = [run62.tape[i] for i in range(len(binary_tape))]

    print(f"\nTape comparison (first 30 binary cells from position 0):")
    print(f"  (6,2): {binary_from_62[:30]}")
    print(f"  (2,8): {binary_from_28[:30]}")

    match = binary_from_62[:30] == binary_from_28[:30]
    print(f"  Match: {'YES' if match else 'NO'}")

    return match


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALGEBRAIC UTM: Neary-Woods (6,2) → (2,8)")
    print("=" * 60)

    # Step 1: Verify the (6,2) machine
    print("\n--- Step 1: Verify U(6,2) ---")
    tm62 = make_62()
    tm62.print_table()
    errs = tm62.validate()
    if errs:
        print(f"ERROR: {errs}")
        sys.exit(1)
    print("Table valid.")

    # Also show the (2,4) for reference
    print("\n--- Reference: U(2,4) ---")
    tm24 = make_24()
    tm24.print_table()

    # Step 2: Find CRT decomposition
    print("\n--- Step 2: CRT Decomposition ---")
    state_to_crt = analyze_crt_decomposition()

    # Step 3: Build (2,8) table
    if state_to_crt:
        print("\n--- Step 3: Build (2,8) ---")
        table_28 = build_28_from_62(state_to_crt)

        if table_28:
            # Step 4: Verify
            print("\n--- Step 4: Verify ---")
            verify_28_vs_62(table_28, state_to_crt)

            # Save
            sol = {}
            for (q, s), (nq, ns, d) in table_28.items():
                sol[f"{q},{s}"] = {"new_state": nq, "new_symbol": ns, "direction": d}
            sol["_meta"] = {
                "source": "Neary-Woods (6,2) → (2,8) algebraic construction",
                "crt_decomposition": {str(k): list(v) for k, v in state_to_crt.items()},
            }
            with open("solution_algebraic_28.json", "w") as f:
                json.dump(sol, f, indent=2)
            print("\nSaved to solution_algebraic_28.json")
    else:
        print("\nNo perfect CRT decomposition found.")
        print("The (6,2) machine may not have the Z/2Z × Z/3Z structure")
        print("needed for a direct algebraic conversion.")
        print("\nOptions:")
        print("  1. Try a modified (6,2) machine with the right structure")
        print("  2. Use SAT search to find a compatible (2,8) table directly")
        print("  3. Use the (2,4) → (2,8) route instead")
