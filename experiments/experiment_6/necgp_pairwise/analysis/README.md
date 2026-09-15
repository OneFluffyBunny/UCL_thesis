# necgp_pairwise analysis scripts

Post-hoc measurements behind `../../RESULTS.md`, section 3. Each script locates
`necgp_pairwise/` from its own path, so it can be run from anywhere. All but
`nand_min.py` read `runs/base/`, made by

    ../../experiment_5/.venv-pypy/Scripts/pypy.exe train.py --seeds 0-4 --tag base

(300,000 generations, `retina_ka2005`/xor, NAND only). Any interpreter works; under
PyPy each takes seconds.

| script | measures | result on `runs/base` |
|---|---|---|
| `nand_min.py K` | minimum NAND count of every function of ≤4 inputs, exhaustive up to K gates (K = 7 takes minutes, 8 does not finish) | XOR, MUX, AND3, NOR 4; XNOR 5; OR3, MAJ3 6; NOR3 7; object detector > 7 |
| `compress_rate.py` | active nodes, genome-adjacent used wires, how often `compress` succeeds (500 tries per snapshot, 234 snapshots) | 36.7/100 active; 6.3% adjacent; 0.62% succeed, 0.26% with both gates active |
| `move_legal.py` | for each non-adjacent used gate-to-gate wire, whether a reorder can make it adjacent without changing the circuit | 16,245 wires; moving either end 53.1%; some reorder 76.7%; reorder within the size cap 70.6% |
| `check_ops.py` | 200k single point mutations of seed 1's final genotype; node types in each seed's final active circuit | NAND→module 4,524, module→NAND 4,812, input 2→5 187; no compress-made call in any final circuit |
| `iface.py` | module interface sizes; active mutation-made calls read only at an output equal to NAND of their first two inputs | (3,2) 563, (4,2) 334, (3,1) 103, (4,1) 65; 2,416 of 7,076 calls (34%) |
