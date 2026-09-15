# necgp_pairwise — nested ECGP with interaction-checked compress

`../necgp/` wraps windows of genome-adjacent nodes into modules whether or not those
nodes are wired together, and in a solved circuit 37.5% of module calls went to such
fake modules (`../RESULTS.md`, section 2). This variant changes `compress` so that
cannot happen. Full mechanism: `ecgp.py`'s module docstring.

## Mechanism

- `compress` proposes exactly two genome-adjacent nodes. The candidate module is built
  and accepted only if its exposed outputs depend on a connection between the two
  (`module_has_interaction`); otherwise nothing changes. `validate()` asserts that no
  fake module exists.
- An existing module can be one of the two nodes, so modules nest, gated by
  `nest_decay` as in `necgp/`.
- `max_module_size` (5) caps the module's flattened NAND count (a body always has two
  slots, so a slot bound would do nothing).
- No module point mutation and no add/remove input/output operators: a module's
  interface is fixed when it is made; changing it means `expand` then `compress`.
  The remaining parameters are `mutation_rate`, `compress_prob`, `expand_prob` and
  `nest_decay`.

## Run

```bash
# search: PyPy, seeds in parallel (~1 minute for 5 seeds); CPython also works, ~5x slower
../../experiment_5/.venv-pypy/Scripts/pypy.exe train.py --seeds 0-4 --tag base
# figures: CPython (matplotlib)
python render.py runs/base
```

The PyPy venv is built by `python ../../experiment_5/setup_pypy.py`. `run.py` is the
same search without history files; `test_train.py` checks that `train.py` equals
`run.py` with snapshots on and off, that PyPy equals CPython, and that every snapshot
replays to its logged score.

**Outputs per seed** (`runs/<tag>/seed<k>/`): `log.csv` (one row per snapshot),
`gates.csv` (per snapshot, one row per NAND or live module: call counts, share of
top-level and flattened gates, truth-table signature, a name such as `XOR` when there
is one, depth, birth generation), `snapshots.jsonl` (the genotype, so any snapshot can
be redrawn), `result.json`. Snapshots are taken at generation 0, every
`--snapshot-interval` (1000) generations, on every improvement and at the end. Column
definitions: `census.py`.

**Figures** (`render.py`): `gate_shares.png` (gate mix over evolution with accuracy),
`stages.png` (six snapshots), `final_decomposition.png`. A module keeps one colour in
all three; the seven most used are coloured and the rest pooled in grey.

`analysis/` holds the post-hoc measurements on `runs/base` reported in `../RESULTS.md`.
