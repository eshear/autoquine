# (32,2) Reversible UTM Search

SAT-based search for a 32-state, 2-symbol reversible Turing machine that
simulates the known (5,5) universal Turing machine.

## Theory

To encode the 5 symbols of the (5,5) machine into binary (F_2), we use
3-bit blocks on the tape (since 2^2 = 4 < 5, but 2^3 = 8 >= 5).

The 32 states arise from a factorisation of a 5-bit register:

    F_2^5  ≅  F_2^3  ⊗  F_2^2

- **F_2^3 (8 values) — Logical State**: 5 macro-states of the base (5,5)
  machine plus 3 reversible boundary/bounce reflectors.
- **F_2^2 (4 values) — Spatial Gauge**: 3 phases tracking head position
  inside a 3-bit macro-block, plus 1 bit for historical sweep direction
  (L-to-R vs R-to-L).

8 × 4 = 32 states.

## Approach

We formulate a SAT/SMT problem whose solutions are transition tables for a
(32,2) reversible TM that faithfully simulates the (5,5) UTM:

1. **Strict reversibility**: the transition function is a bijection on
   (state, symbol) pairs — every configuration has exactly one predecessor.
2. **Simulation correctness**: macro-steps of the (32,2) machine correspond
   to single steps of the (5,5) machine on 3-bit-encoded tapes.

## Usage

```
python3 utm_search.py              # run the SAT search
python3 utm_search.py --verify     # verify a found solution
python3 -m pytest test_utm.py -v   # run tests
```

## Requirements

```
pip install python-sat z3-solver
```
