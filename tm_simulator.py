#!/usr/bin/env python3
"""
Generic Turing Machine simulator.

Supports any (q, s) machine:
  - table: {(state, symbol): (new_state, new_symbol, direction)}
  - direction: -1 (L) or +1 (R)
  - tape: defaultdict with configurable blank symbol
"""

from collections import defaultdict


class TM:
    """Generic Turing Machine."""

    def __init__(self, table, n_states, n_symbols, blank=0,
                 name="TM", state_names=None, symbol_names=None):
        self.table = dict(table)
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.blank = blank
        self.name = name
        self.state_names = state_names or {i: str(i) for i in range(n_states)}
        self.symbol_names = symbol_names or {i: str(i) for i in range(n_symbols)}

    def validate(self):
        """Check table is complete and well-formed."""
        errors = []
        for q in range(self.n_states):
            for s in range(self.n_symbols):
                if (q, s) not in self.table:
                    errors.append(f"Missing entry ({q}, {s})")
                else:
                    nq, ns, d = self.table[(q, s)]
                    if nq < 0 or nq >= self.n_states:
                        errors.append(f"({q},{s}): bad new_state {nq}")
                    if ns < 0 or ns >= self.n_symbols:
                        errors.append(f"({q},{s}): bad new_symbol {ns}")
                    if d not in (-1, 1):
                        errors.append(f"({q},{s}): bad direction {d}")
        return errors

    def print_table(self):
        """Pretty-print the transition table."""
        dn = {-1: 'L', 1: 'R'}
        print(f"\n{self.name} ({self.n_states},{self.n_symbols}) transition table:")
        print(f"  {'St':>6} {'Sym':>4} -> {'Wr':>4} {'Dir':>4} {'Nxt':>6}")
        print(f"  {'-' * 32}")
        for q in range(self.n_states):
            for s in range(self.n_symbols):
                if (q, s) in self.table:
                    nq, ns, d = self.table[(q, s)]
                    print(f"  {self.state_names[q]:>6} {self.symbol_names[s]:>4} -> "
                          f"{self.symbol_names[ns]:>4} {dn[d]:>4} {self.state_names[nq]:>6}")
            print()


class TMRun:
    """A running TM instance with tape and head."""

    def __init__(self, tm, tape=None, head=0, state=0):
        self.tm = tm
        self.tape = defaultdict(lambda: tm.blank)
        if tape is not None:
            for i, v in enumerate(tape):
                self.tape[i] = v
        self.head = head
        self.state = state
        self.steps = 0
        self.halted = False

    def step(self):
        """Execute one step. Returns True if a transition was found."""
        key = (self.state, self.tape[self.head])
        if key not in self.tm.table:
            self.halted = True
            return False
        nq, ns, d = self.tm.table[key]
        self.tape[self.head] = ns
        self.state = nq
        self.head += d
        self.steps += 1
        return True

    def run(self, max_steps=10000, trace=False):
        """Run until halt or max_steps."""
        if trace:
            self._print_config()
        for _ in range(max_steps):
            if not self.step():
                if trace:
                    print(f"  HALTED at step {self.steps}")
                return self.steps
            if trace:
                self._print_config()
        return self.steps

    def get_tape_segment(self, lo=None, hi=None):
        """Get a contiguous tape segment as a list."""
        if lo is None or hi is None:
            positions = [k for k, v in self.tape.items() if v != self.tm.blank]
            if not positions:
                return [], 0, 0
            lo = min(positions) if lo is None else lo
            hi = max(positions) + 1 if hi is None else hi
        return [self.tape[i] for i in range(lo, hi)], lo, hi

    def tape_str(self, lo=None, hi=None, width=40):
        """Compact tape string with head marker."""
        seg, lo, hi = self.get_tape_segment(lo, hi)
        # Ensure head is visible
        if self.head < lo:
            lo = self.head
        if self.head >= hi:
            hi = self.head + 1
        seg = [self.tape[i] for i in range(lo, hi)]
        parts = []
        for i, pos in enumerate(range(lo, hi)):
            sym = self.tm.symbol_names.get(seg[i], str(seg[i]))
            if pos == self.head:
                parts.append(f"[{sym}]")
            else:
                parts.append(f" {sym} ")
        return ''.join(parts)

    def _print_config(self):
        sn = self.tm.state_names.get(self.state, str(self.state))
        print(f"  t={self.steps:5d} q={sn:>4} h={self.head:4d}  {self.tape_str()}")


def check_morita_reversibility(table, n_states, n_symbols):
    """Check Morita's reversibility condition.

    For each (new_state, direction) pair, all transitions producing
    that output must write distinct symbols.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for (q, s), (nq, ns, d) in table.items():
        groups[(nq, d)].append((q, s, ns))

    for (nq, d), entries in groups.items():
        written = [e[2] for e in entries]
        if len(written) != len(set(written)):
            return False, (nq, d), entries
    return True, None, None


def check_local_injectivity(table, n_states, n_symbols):
    """Check if the transition function is locally injective.

    (state, symbol) -> (new_state, new_symbol, direction) is injective.
    """
    outputs = set()
    for (q, s), (nq, ns, d) in table.items():
        out = (nq, ns, d)
        if out in outputs:
            return False
        outputs.add(out)
    return True


if __name__ == "__main__":
    # Self-test with a simple counter TM
    # 2-state 2-symbol increment: counts in unary
    table = {
        (0, 0): (1, 1, 1),   # q0, blank -> write 1, move R, halt (q1)
        (0, 1): (0, 1, 1),   # q0, 1 -> keep 1, move R, stay q0
        (1, 0): (1, 0, -1),  # q1, blank -> stay, move L (halt-ish)
        (1, 1): (1, 1, -1),  # q1, 1 -> keep, move L
    }
    tm = TM(table, 2, 2, name="Counter")
    tm.print_table()

    run = TMRun(tm, tape=[1, 1, 1, 0], head=0, state=0)
    steps = run.run(max_steps=20, trace=True)
    print(f"Ran {steps} steps")
