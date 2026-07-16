"""Reload a saved best-DNA checkpoint and (re)draw its brain.

Usage:
    python visualize_ckpt.py runs/copy_seed0_best_dna.eqx
    python visualize_ckpt.py runs/retina_seed0_best_dna.eqx --n-hidden 20 -K 4

The architecture flags must match those the checkpoint was trained with (they are
not stored in the .eqx of leaves alone). Pass the same --n-in/--n-hidden/-K/etc.
"""

from __future__ import annotations

import argparse
import os

import jax.random as jr
import equinox as eqx

from model import Genome
from config import build_parser, build_brain_config
from visualize import visualize_brain, brain_stats


def main():
    # reuse the architecture flags from config, plus a checkpoint path + threshold
    parser = build_parser()  # provides --no-open and all architecture flags
    parser.add_argument("ckpt", help="path to a *_best_dna.eqx checkpoint")
    parser.add_argument("--save", default=None, help="output PNG path (default: alongside ckpt)")
    args = parser.parse_args()
    open_after = not args.no_open

    brain_cfg = build_brain_config(args)

    # build a template with the right structure, then load leaves into it
    template = Genome.init(jr.PRNGKey(0), brain_cfg)
    genome = eqx.tree_deserialise_leaves(args.ckpt, template)

    save = args.save or os.path.splitext(args.ckpt)[0] + "_viz.png"
    st = brain_stats(genome, brain_cfg, args.prune_threshold)
    print(f"loaded {args.ckpt} | edges {st['n_edges']}/{st['max_edges']} | "
          f"density {st['density']:.1f}% | hidden type counts {st['type_counts']}")
    visualize_brain(genome, brain_cfg, args.prune_threshold, save,
                    title=f"checkpoint: {os.path.basename(args.ckpt)}",
                    open_after=open_after)


if __name__ == "__main__":
    main()
