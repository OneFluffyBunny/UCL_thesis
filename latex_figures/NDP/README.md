# NDP figures

Copied from the standalone NDP work, now imported under `NDP/` (git subtree,
2026-09-15). The trained models they were drawn from are in `NDP/saved_models/`,
which is gitignored and exists only on the author's laptop.

| file | source |
|---|---|
| `KA/brains_grid_leftright_matched.png`, `KA/progress_fg_vs_mvg_matched.png` (2026-09-14) | `NDP/experiments_paper/retina/matched_figures.py grid` / `progression`; documented in `NDP/experiments_paper/retina/RESULTS.md` (budget-matched FG vs MVG, 5 runs per arm) |
| `KA/FG_brain.png`, `KA/communities_FG_1786033855.png`, `KA/leftright_optimal_FG_1786033855.png` (2026-08-07/08) | KA retina FG run `saved_models/1786033855` |
| `KA/MVG_brain.png`, `KA/communities_MVG_1786053806.png`, `KA/leftright_optimal_MVG_1786053806.png` (2026-08-07/08) | KA retina MVG run `saved_models/1786053806` |
| `CartPole/graph_best.png`, `CartPole/growth_stages.png` (2026-08-05) | a CartPole run's `graph_best.png` (`train.py`) and `growth_stages.py`; the run id was not recorded (`RESULTS.md` quotes CartPole run `1785945244`, unverified as the source) |

The FG/MVG `*_brain.png` files are assumed to be the same two runs as the community
figures; that pairing was not recorded when they were copied.
