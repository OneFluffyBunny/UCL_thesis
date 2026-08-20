# Experiment 6 — nested modules: does evolution keep and reuse them?

Experiment 4's ECGP (`../experiment_4/`) has `compress`/`expand`: a genotype can fold
a handful of active nodes into a callable module and later inline it back. But
**ECGP explicitly forbids nesting** — a module can never contain another module
(`ecgp.py`'s `compress`, paper-verbatim guard). That is a real ceiling on what
"module re-use" can mean: a module can be called many times, but modules can never
build on top of each other, so there is no way for evolution to discover a small
useful part, then a larger part made *of* those parts, and so on.

**The point of experiment 6 is to build a nested variant of ECGP** — something that
lets a genotype construct increasingly intricate, hierarchical modules — and use it
to ask the question experiment 4 cannot: when nesting is available *and* its
complexity is priced (so it cannot just explode), does evolution actually keep
useful modules around, call them repeatedly, and build bigger structures out of
them? That is the emergence-of-modularity question this whole thesis is about,
aimed at the one mechanism (hierarchical composition) the other experiments don't
have.

## Status

**SMCGP is implemented, tested, and verified to search correctly** (see
`RESULTS.md`). It is not what the final nested-module system will be — see
"Where this goes next" below — but it is a real, working alternative
developmental mechanism that grows structure without an explicit module
registry, and a useful reference point while the real target is designed.

## Why SMCGP first

Two published mechanisms produce something like nested/hierarchical structure in
CGP, and only one of them turned out to be buildable from a paper actually in hand:

- **Modular CGP (MCGP)** — Walker & Miller's own direct extension: it lets a
  module contain modules, unboundedly, which is the closest existing match to
  "nested modules." **Not implemented here.** Its only two primary sources
  (J. A. Walker's 2008 PhD thesis; Ch. 3 of Miller (ed.), *Cartesian Genetic
  Programming*, Springer 2011) were both unreachable this session — the thesis
  isn't in White Rose's open repository, and the book chapter sits behind
  Springer's login wall. Building MCGP specifically (Walker & Miller's exact
  nesting/complexity-control mechanism) is blocked until one of those is sourced
  through the library.

- **Self-Modifying CGP (SMCGP)** — Harding, Miller & Banzhaf, CEC 2009 (local copy:
  `../../papers/Harding_Miller_Banzhaf_2009_Self_Modifying_CGP_Parity.pdf`, a free,
  author-hosted PDF, read in full). A *developmental* mechanism: the genotype
  contains ordinary computational gates **and** thirteen graph-rewriting
  operators (duplicate, delete, move, change-function, ...), and a shared,
  length-capped "To Do" list applies them iteratively before each evaluation,
  regrowing the phenotype from the genotype every time. This **is** fully
  specified in a source actually obtained — see `PAPER_SPEC.md` — so it's what
  got built.

**SMCGP is not the same claim as "nested modules," and that gap matters:** SMCGP
has no persistent, named, callable module — nothing plays the role of ECGP's
module registry. What grows is a flat, self-rewriting graph; a repeated
sub-structure exists only because a DUP/MOV/DU3 operator keeps re-inserting a
copy of it (relative addressing is exactly what makes a duplicate remain valid
wherever it lands — see `PAPER_SPEC.md`'s "Representation" section). So SMCGP can
answer a *related* question — does self-modifying growth converge on reusable
sub-graphs at all — but not the literal one this experiment exists for (does a
genotype keep, call, and build upon **named** modules under a nesting-depth
cost). Treat the two as separate mechanisms being compared, not one thing wearing
two names.

## Where this goes next

**Status update (2026-08-19): the nested-ECGP target is now built** — see
`necgp/` and `RESULTS.md`'s "necgp built" entry. MCGP's own answer to "how do you
stop nesting from exploding" is still locked behind the two unreachable sources
above, so this is **our own extension**, not a reproduction of anyone's paper: it
relaxes `ecgp.py`'s `compress` no-nesting guard so a module *can* contain modules,
gated by a decaying probability, `nest_decay ** (depth - 1)`, starting at 0.5 (a
new `--nest-decay` CLI flag) — deeper nesting stays possible but costs likelihood,
the same way a real mutation-pressure parsimony term would. A first smoke run
(100k generations, NAND-only, retina/xor) solved the task and produced real
depth-2/depth-3 modules, including one module reusing the same submodule twice
inside its own body — see `RESULTS.md` for the full readout.

This is v1, explicitly scoped down in one place (`necgp/ecgp.py`'s module
docstring): the four module-interface operators (add/remove input/output) refuse
to touch a module that is nested-into by another module, rather than repairing
call sites inside bodies. Alternatives to the decay curve (an explicit
MDL/description-length penalty, a hard depth cap with an evolvable cap value)
remain open if trial-and-error on `--nest-decay` and the operator set doesn't turn
out to be enough — nothing about the decay-curve choice is locked in.

## Layout

Self-contained, following `experiment_4`/`experiment_5`'s own fork convention
(each experiment owns its files rather than importing across directories, so one
experiment's changes can never silently move another's frozen results). Two
mechanisms live here side by side, each in its own scope:

| File | What it is |
|---|---|
| `smcgp.py` | SMCGP: genotype, development (self-modification), mutation, bit-parallel evaluation. |
| `gates.py` | SMCGP's computational (non-self-modifying) boolean function set — restricted (AND/OR/NAND/NOR) or full (BF0..BF15). |
| `tasks.py` | SMCGP's even-parity curriculum (the paper's own task). |
| `config.py` | SMCGP's CLI flags / `RunConfig`. Paper-verbatim defaults are marked; see `PAPER_SPEC.md`. |
| `train.py` | SMCGP's (1+4) evolutionary strategy loop, with checkpoint/resume and CSV logging (same output-file convention as experiment_4/5). |
| `test_smcgp.py` | SMCGP correctness tests: relative-addressing edge cases, each hand-checkable operator, an end-to-end smoke run. |
| `PAPER_SPEC.md` | SMCGP's full spec, every claim tagged `[verbatim]` / `[inferred]` / `[our choice]`. Read this before trusting any numeric default. |
| `necgp/` | The actual nested-ECGP target — see below. |
| `RESULTS.md` | The lab notebook for both mechanisms. |

Run SMCGP: `conda run -n lndp python train.py --max-inputs 6 --max-evals 300000`
(the paper's own scale, `--max-inputs 20 --max-evals 10000000`, is hours of
pure-Python work — see `config.py`).

### `necgp/` — nested ECGP

A further self-contained fork *inside* experiment 6, one directory deeper than
`experiment_4`/`experiment_5` so `tasks.py`'s path back to `kashtan_alon/` climbs
one extra level (see its own header comment):

| File | What it is |
|---|---|
| `cgp.py`, `gates.py` | Copied unmodified from `experiment_4/` — the CGP baseline `ecgp.py` flattens onto, and the boolean gate set. |
| `tasks.py` | The KA retina task, adapted from `experiment_4/tasks.py` for the extra directory depth. |
| `ecgp.py` | Forked from `experiment_4/ecgp.py`, `compress`'s no-nesting abort replaced by a depth-gated accept (`Params.nest_decay`). Every place the fork differs from plain ECGP is tagged `EXTENDED` in-line; the module docstring at the top is the map. |
| `smoke.py` | A plain (1+4) ES driver, fixed generation budget — no checkpointing/CSV, not `experiment_4/train.py` — for looking at what nesting does. `--nest-decay`, `--gates`, `--task`/`--operation` are the knobs most worth trying next. |
| `visualize.py` | Fork of `experiment_4/visualize.py`, unflattened (module = one box) renderer only, every box labelled with a `|x` **nesting factor** (`|1` primitive, `|2` a module of primitives, `|3` a module nesting a `|2`, ...) — the thing plain ECGP's drawing never needed because its modules were always exactly one level deep. |
| `stage_run.py` | Run-to-solution driver (stops at first solve, not a fixed budget): runs the given `--nest-decay` and a `--nest-decay 0.0` twin back to back on the same seed for a direct generations/wall-clock comparison, then draws the nested run's own history as a 2x3 stage sheet via `visualize.stage_grid`. |
| `test_necgp.py` | Correctness tests specific to the nesting extension: structural invariants under heavy mutation, `evaluate()` vs. `flatten()`, the decay actually gating depth, `expand` undoing exactly one level, the nested-into interface-operator guard, transitive pruning. |

Run: `conda run -n lndp python smoke.py --gates nand --nest-decay 0.5`, or
`conda run -n lndp python stage_run.py --gates nand --nest-decay 0.5` for a
run-to-solution + stage-sheet + nested-vs-flat comparison (see `RESULTS.md`'s
"run-to-solution, NAND-only" entry for what the first run of this found).

## Speed

Pure Python, bit-parallel truth-table masks (one Python int per wire, all
`2**n_in` patterns evaluated as one bitwise op), the same convention
`experiment_4`/`experiment_5` use — see `CLAUDE.md`'s established fact that this
caps out around 20 inputs, which is exactly the paper's own ceiling. Development
(the self-modification iterations) is pure graph-structure bookkeeping with no
per-pattern cost, so it stays cheap even as `--max-inputs` grows; only the
*final*, fully-developed phenotype of each test case is evaluated exhaustively.
No PyPy port yet — `experiment_5/bench.py --crossover`'s finding (PyPy wins only
below ~14 inputs) would need re-measuring here since development's node-list
churn (insert/delete on plain Python lists) is a different workload from
experiment_4/5's fixed-length integer-array genotype.
