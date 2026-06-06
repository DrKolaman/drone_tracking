# Tracker tests

Regression/TDD safety net for the stabilized DINOv3 + motion tracker, so the code
can be optimized/reordered later and still behave the same on `data/source.mp4`.

## Layers
- **Fast unit tests** (no video, no DINOv3, milliseconds): `test_registration.py`,
  `test_target_memory.py`, `test_bytetrack_shim.py`, `test_detect.py`. These are
  the real red-green-refactor surface for the individual modules.
- **Characterization** (`test_characterization.py`, `@slow`): runs the full
  tracker on the clip and asserts behaviour *bands* — no far false jumps
  (≤50 px adjacent), the f120–200 region stays covered (the old f124 dropout),
  coverage ≥68%, only known states. These catch real regressions.
- **Golden master** (`test_golden.py`, `@slow`): compares a fresh run to
  `golden/track_ref.csv` (same frame count, state match ≥85%, box centre ≤15 px).

## Run
```bash
# fast (default; safe to run constantly)
python3 -m pytest -m "not slow"

# slow (needs a GPU + a Hugging Face token for the gated DINOv3 model)
HF_TOKEN=<token> python3 -m pytest -m slow
```

## Regenerate the golden
Only when you INTENTIONALLY change tracker behaviour:
```bash
REGEN_GOLDEN=1 HF_TOKEN=<token> python3 -m pytest tests/test_golden.py -m slow
```

## Note on tolerances
DINOv3 GPU fp16 inference is **not bit-exact run-to-run**, so the slow tests use
tolerance bands rather than exact equality (observed run-to-run drift: max-jump
~15→31 px, coverage ~74→83%). The bands are set to catch genuine regressions
(e.g. the 142 px far-jump, a lost segment) while tolerating that noise. For
stricter reproducibility, run the embedder in fp32 / deterministic mode.
