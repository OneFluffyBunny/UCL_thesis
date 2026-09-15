# necgp_pairwise analysis scripts

One-off measurements behind the module-acquisition section of `add_to_latex.md`
(2026-09-15) and the matching `../../RESULTS.md` entry. Promoted from a session
scratch folder so the numbers stay reproducible; code unchanged except that each
script now finds `necgp_pairwise/` from its own location.

They read `runs/base/` (gitignored), made by the README's own command:

    ../../experiment_5/.venv-pypy/Scripts/pypy.exe train.py --seeds 0-4 --tag base

(git commit `30a5748`, 300k generations, `retina_ka2005`/xor, NAND only). The
2026-09-15 rerun of every script from the repo root reproduced the numbers below exactly.
Any interpreter works; PyPy takes seconds.

| script | measures | result on `runs/base` |
|---|---|---|
| `nand_min.py K` | minimum NAND count of every function of <=4 inputs, exhaustive up to K gates (K=7 takes minutes, 8 does not finish) | XOR/MUX/AND3/NOR 4, XNOR 5, OR3/MAJ3 6, NOR3 7, KA object >7 |
| `compress_rate.py` | active nodes, genome-adjacent used wires, how often `compress` succeeds (500 tries per snapshot, 234 snapshots) | 36.7/100 active; 6.3% adjacent; 0.62% succeed, 0.26% with both gates active |
| `move_legal.py` | for each non-adjacent used gate->gate wire, can a reorder make it adjacent without changing the circuit | 16,245 wires; move either end 53.1%; some reorder 76.7%; reorder within the size cap 70.6% |
| `check_ops.py` | 200k single point mutations of seed 1's final genotype; node types in every seed's final active circuit | NAND->module 4,524, module->NAND 4,812, input 2->5 187; 0 compress-made calls in any final circuit |
| `iface.py` | module interface sizes; active mutation-made calls whose read outputs are just NAND of wires 0,1 | (3,2) 563, (4,2) 334, (3,1) 103, (4,1) 65; 2,416 of 7,076 calls (34%) |
