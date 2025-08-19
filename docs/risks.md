# Risk Register

This risk register enumerates the primary technical and operational
hazards facing the dual‑book trading lab and proposes monitoring and
mitigation strategies.  Where appropriate, external guidance is
cited.

## Data leakage

**Detection:**

* Continuously validate that training, validation and test splits
  respect temporal ordering and purging/embargo rules.  Use
  combinatorial purged cross‑validation (CPCV) to detect and quantify
  leakage.
* In CI, run the supplied walkforward and deterministic splitting
  tests to ensure no future information contaminates the training set.

**Mitigation:**

* Apply embargoes when forming rolling windows to ensure that
  overlapping samples do not introduce look‑ahead bias.
* Shift realised volatility windows and other trailing features by one
  bar to avoid using information from the current bar.
* Use separate scalers fit on the training split and apply them to
  validation and test splits.

**Fallback:** If leakage is detected despite these measures, freeze
model deployment and revert to the last known good snapshot until the
source of leakage is corrected.

## Overfitting

**Detection:** Monitor the gap between training and validation loss
curves.  Sudden divergence indicates memorisation rather than
generalisation.  Evaluate models on held‑out CPCV folds.

**Mitigation:** Use dropout, weight decay and early stopping during
training.  Employ uncertainty‑weighted loss balancing and gradient
normalisation to stabilise multi‑task optimisation.  Expand the
feature set to include noise‑robust indicators rather than chasing
idiosyncratic patterns.

**Fallback:** Reduce model complexity by decreasing the number of
layers or hidden units, or revert to ridge‑based baselines when
overfitting cannot be controlled.

## Adaptive Computation Time (ACT) instability

**Detection:** Monitor the distribution of ACT halting steps and the
ponder cost during training.  If the model consistently triggers
either the first or last possible segment, the Q‑head may be
collapsing.

**Mitigation:** Tune the ponder cost and maximum steps; initialise
the Q‑head biases to encourage exploration; and clip gradients on
the Q‑head.  If instability persists, disable ACT and fall back to a
fixed number of segments.

**Fallback:** Disable ACT entirely and rely on deep supervision with
a fixed segment count.

## Regulatory mis‑specification (SSR/LULD)

**Detection:** Compare the simulator behaviour against authoritative
sources.  For example, SEC Rule 201 stipulates that once the short
sale restriction (SSR) is triggered by a 10 % intraday drop, it
remains in effect through the end of the following trading day【515680207816213†L213-L218】.
Similarly, the national Limit Up–Limit Down (LULD) price bands are
doubled during the last 25 minutes of regular trading for Tier 1
securities and Tier 2 securities below $3【457522738594840†L70-L71】.  Unit
tests should assert that these behaviours are implemented exactly.

**Mitigation:** Implement stateful SSR logic that persists the
restriction across days, enforce uptick‑only execution, and double
LULD band widths near the close.  Verify these rules against SEC
publications and industry guidance.  Update documentation when
regulations change.

**Fallback:** If rule changes are announced, temporarily disable the
affected constraint in simulation until it can be recoded and
validated.

## Determinism drift

**Detection:** Run deterministic tests in CI to ensure that repeated
runs with the same seed produce identical outputs.  Monitor hash
digests of intermediate artefacts.

**Mitigation:** Seed all pseudo‑random number generators at the
framework, library and numpy levels.  Avoid sources of nondeterminism
such as parallel data loaders without pinned workers.  Fix the
ordering of dictionary iterations and sorting operations.

**Fallback:** If determinism cannot be guaranteed on a given
accelerator, pin the model to CPUs or use an alternative compute
backend.

## Latency overruns

**Detection:** Measure per‑segment forward pass latency in unit tests
and profile the end‑to‑end inference pipeline.  Set budgets for the
H‑ and L‑modules separately.

**Mitigation:** Cap model size at approximately 27 million
parameters, use mixed‑precision inference, and batch minute‑level
tokens efficiently.  If necessary, prune attention heads or reduce
hidden dimensionality.

**Fallback:** If the system cannot meet the latency target on
available hardware, fall back to the ridge‑based baseline until
hardware upgrades are obtained or the model is simplified.