"""Oracle / theoretical-capacity probe for experiment 1.

Given a task whose target boolean function we KNOW, we can hand-wire a brain that
solves it perfectly, and then ask whether the cell-type encoding can *represent*
that brain. Three stages:

  1. Build an oracle wiring (w, b) by hand and run it through the REAL inference
     dynamics -> confirms the brain (the inference) can score 1.0   ["brain ceiling"]
  2. OLS-fit the shared connection rule `g` to those oracle weights, with the
     neuron features held fixed (closed-form least squares on g's final layer).
  3. Plug the fitted g back into a genome ("oracle DNA") and run the real
     forward pass to measure the exact score this DNA achieves   ["encoding ceiling"]

If stage 3 scores ~1.0, a perfect genome EXISTS -> our training failure is
reachability/landscape, not representability. If it can't, the perfect brain is
not in the set the encoding can grow -> loosen the encoding.

This is a deliberately simple implementation (closed-form OLS on the last layer
with fixed random hidden features). Future work: full non-linear fit of g (and
the embeddings) to measure the true theoretical capacity of the framework.

Run:  python oracle.py            # left task
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import tasks
from model import BrainConfig, Genome, _role_mask

ON = 0.9          # "present" synapse weight (must be < 1; g's tanh asymptotes at 1)
GAIN_OK = True    # see logic in build_oracle_left


# ---------------------------------------------------------------------------
# Stage 1: hand-wired oracle for `left` = (p0&p1) | (p2&p3)
# ---------------------------------------------------------------------------

def left_config(g_width: int = 16) -> BrainConfig:
    # 4 inputs, 2 hidden (one per AND-detector type), 1 output, K=2 types.
    # g_width defaults to the training value (16) so the capacity matches the DNA
    # we actually evolve -- a wider g here would make the representability test
    # nonsensical (it would prove a perfect genome exists in a *different* family).
    return BrainConfig(n_in=4, n_hidden=2, n_out=1, n_types=2,
                       g_width=g_width, g_depth=1, rnn_iters=8, activation=jnp.tanh)


def build_oracle_left(cfg: BrainConfig):
    """Return (w_u target at signature level (U,U), type_bias (K+2,)).

    Signature order matches Genome.unique_structure: [in0..in3, typeA, typeB, out].
    Logic (bipolar inputs in {-1,+1}, output thresholded at 0):
      A = p0 AND p1 : preact = ON*x0 + ON*x1 + bias_A,  bias_A = -1.0
                      both-on -> +0.8 (>0); else <= -1.0  -> sign encodes the AND
      B = p2 AND p3 : same on x2,x3
      out = A OR B  : preact = ON*A + ON*B + bias_out,   bias_out = 0.5
                      both-false -> negative; either-true -> positive
    """
    U = cfg.n_in + cfg.n_types + cfg.n_out          # 4 + 2 + 1 = 7
    iA = cfg.n_in + 0                               # typeA signature row = 4
    iB = cfg.n_in + 1                               # typeB signature row = 5
    iout = cfg.n_in + cfg.n_types                   # out signature row   = 6

    w_u = jnp.zeros((U, U))
    w_u = w_u.at[0, iA].set(ON).at[1, iA].set(ON)   # in0,in1 -> A
    w_u = w_u.at[2, iB].set(ON).at[3, iB].set(ON)   # in2,in3 -> B
    w_u = w_u.at[iA, iout].set(ON).at[iB, iout].set(ON)  # A,B -> out

    # type_bias rows: [0..K-1 hidden types, K -> input, K+1 -> output]
    type_bias = jnp.array([-1.0, -1.0, 0.0, 0.5])
    return w_u, type_bias


# ---------------------------------------------------------------------------
# Stage 1: hand-wired oracle for `retina` AND =
#   [(p0&p1)|(p2&p3)] AND [(p4&p5)|(p6&p7)]
# ---------------------------------------------------------------------------

def retina_config(g_width: int = 16, n_types: int = 6) -> BrainConfig:
    # 8 inputs, 1 output. Needs SIX distinct hidden cell-types (see below), so
    # n_types defaults to 6 -- the *minimum* for representability. Training ran at
    # K=4, which is why the perfect brain is not in that family. n_hidden=n_types
    # (abundance=0 -> one neuron per type). rnn_iters must span the 3-layer depth.
    return BrainConfig(n_in=8, n_hidden=n_types, n_out=1, n_types=n_types,
                       g_width=g_width, g_depth=1, rnn_iters=8, activation=jnp.tanh)


def build_oracle_retina(cfg: BrainConfig):
    """Return (w_u (U,U), type_bias (K+2,)) for retina AND.

    Three computational layers (bipolar inputs in {-1,+1}, output thresholded 0):
      L1 detectors (bias -1.0, AND):  A=p0&p1  B=p2&p3  C=p4&p5  D=p6&p7
      L2 combiners (bias +0.8, OR):   Lft=A|B  Rgt=C|D
      out          (bias -0.62, AND): out=Lft & Rgt
    Biases account for tanh attenuation through 3 layers: a stronger combiner bias
    (+0.8) lifts the weak single-detector "on" state clear of 0, and the output AND
    threshold (-0.62) sits between the min both-on and max one-on preactivations.
    Requires 6 distinct hidden types because same-type hidden neurons are clones
    (position-blind), yet A,B,C,D each read a *different* input pair and Lft,Rgt
    read *different* detectors. (A,B,C,D,Lft,Rgt) = types 0..5.
    """
    assert cfg.n_types >= 6, "retina AND needs >= 6 hidden cell-types to be representable"
    U = cfg.n_in + cfg.n_types + cfg.n_out
    h = cfg.n_in                                    # first hidden-type signature row
    A, B, C, D, Lft, Rgt = h + 0, h + 1, h + 2, h + 3, h + 4, h + 5
    iout = cfg.n_in + cfg.n_types

    w_u = jnp.zeros((U, U))
    # L1: input pairs -> detectors
    w_u = w_u.at[0, A].set(ON).at[1, A].set(ON)
    w_u = w_u.at[2, B].set(ON).at[3, B].set(ON)
    w_u = w_u.at[4, C].set(ON).at[5, C].set(ON)
    w_u = w_u.at[6, D].set(ON).at[7, D].set(ON)
    # L2: detectors -> combiners
    w_u = w_u.at[A, Lft].set(ON).at[B, Lft].set(ON)
    w_u = w_u.at[C, Rgt].set(ON).at[D, Rgt].set(ON)
    # out: combiners -> output (final AND)
    w_u = w_u.at[Lft, iout].set(ON).at[Rgt, iout].set(ON)

    # type_bias: [A,B,C,D detectors=-1.0, Lft,Rgt combiners=+0.8, in=0.0, out=-0.62]
    hid_bias = [-1.0, -1.0, -1.0, -1.0, 0.8, 0.8] + [0.0] * (cfg.n_types - 6)
    type_bias = jnp.array(hid_bias + [0.0, -0.62])
    return w_u, type_bias


# ---------------------------------------------------------------------------
# Inference helpers (replicate Genome.forward dynamics for an explicit w, b)
# ---------------------------------------------------------------------------

def run_dynamics(w, b, X_enc, cfg):
    """Run the synchronous recurrent pass for an explicit (w, b) over a batch."""
    def one(obs):
        def step(a, _):
            a = a.at[:cfg.n_in].set(obs)
            a = cfg.activation(a @ w + b)
            return a, None
        a0 = jnp.zeros((cfg.n_total,))
        a, _ = jax.lax.scan(step, a0, None, cfg.rnn_iters)
        return a[-cfg.n_out:]
    return jax.vmap(one)(X_enc)


def balanced_accuracy(preds, y):
    correct = (preds == y).astype(jnp.float32)
    yf = y.astype(jnp.float32)
    pos = jnp.sum(correct * yf) / (jnp.sum(yf) + 1e-8)
    neg = jnp.sum(correct * (1 - yf)) / (jnp.sum(1 - yf) + 1e-8)
    return float(0.5 * (pos + neg))


# ---------------------------------------------------------------------------
# Stage 2: OLS fit of g to the oracle weights (features held fixed)
# ---------------------------------------------------------------------------

def _g_hidden_features(g, X):
    """Output of g's hidden stack (everything before the final linear layer)."""
    h = X
    for layer in g.layers[:-1]:
        h = jax.nn.relu(layer(h))     # nn.MLP default activation is relu
    return h


def fit_g_ols(base_genome, cfg, w_u_target):
    """Closed-form OLS: keep g's hidden layers fixed, solve the final linear
    layer so that tanh(g(pair)) matches the oracle weights on allowed pairs.
    Returns (g_fit, max_abs_residual)."""
    uf, _, _ = base_genome.unique_structure(cfg)
    U = uf.shape[0]

    # all signature pairs [feat_i, feat_j]
    fi = jnp.broadcast_to(uf[:, None, :], (U, U, cfg.feat_dim))
    fj = jnp.broadcast_to(uf[None, :, :], (U, U, cfg.feat_dim))
    pairs = jnp.concatenate([fi, fj], axis=-1).reshape(U * U, 2 * cfg.feat_dim)

    # only fit pairs that survive the role mask (others are zeroed anyway)
    role = _signature_role_mask(cfg).reshape(U * U)
    target = w_u_target.reshape(U * U)

    X = pairs[role]
    t = target[role]
    z = jnp.arctanh(jnp.clip(t, -0.999, 0.999))      # undo g's final tanh

    phi = jax.vmap(lambda x: _g_hidden_features(base_genome.g, x))(X)   # (M, width)
    phi_aug = jnp.concatenate([phi, jnp.ones((phi.shape[0], 1))], axis=1)  # +bias col
    theta, *_ = jnp.linalg.lstsq(phi_aug, z, rcond=None)                  # (width+1,)

    width = phi.shape[1]
    new_w = theta[:width].reshape(1, width)
    new_b = theta[width:width + 1]
    g_fit = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                        base_genome.g, (new_w, new_b))

    # report fit quality on the allowed pairs (post-tanh)
    pred = jax.vmap(lambda x: jnp.tanh(g_fit.layers[-1](_g_hidden_features(g_fit, x))[0]))(X)
    resid = float(jnp.max(jnp.abs(pred - t)))
    return g_fit, resid


def _signature_role_mask(cfg: BrainConfig):
    """(U,U) bool: allowed signature pairs (IH | HH | HO)."""
    U = cfg.n_in + cfg.n_types + cfg.n_out
    idx = jnp.arange(U)
    is_in = idx < cfg.n_in
    is_hid = (idx >= cfg.n_in) & (idx < cfg.n_in + cfg.n_types)
    is_out = idx >= cfg.n_in + cfg.n_types
    ih = is_in[:, None] & is_hid[None, :]
    hh = is_hid[:, None] & is_hid[None, :]
    ho = is_hid[:, None] & is_out[None, :]
    return ih | hh | ho


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="oracle / capacity probe (left | retina AND)")
    p.add_argument("--task", choices=["left", "retina"], default="left")
    p.add_argument("--g-width", type=int, default=16,
                   help="hidden width of the connection-rule g (default 16 = training value)")
    p.add_argument("--n-types", type=int, default=None,
                   help="retina only: hidden cell-types (default 6 = minimum representable; "
                        "set 4 to reproduce the training family that cannot represent it)")
    args = p.parse_args()

    if args.task == "left":
        cfg = left_config(g_width=args.g_width)
        w_u_target, type_bias = build_oracle_left(cfg)
        op = "xor"                               # irrelevant for `left`
    else:
        cfg = retina_config(g_width=args.g_width,
                            n_types=args.n_types if args.n_types is not None else 6)
        w_u_target, type_bias = build_oracle_retina(cfg)
        op = "and"

    print(f"task = {args.task} | op = {op} | g_width = {cfg.g_width} | K = {cfg.n_types}")
    X = tasks.all_binary_inputs(cfg.n_in)
    X_enc = X * 2.0 - 1.0                       # bipolar encoding (training default)
    y = tasks.targets(args.task, op, X)

    # a base genome supplies the fixed neuron features + the g to be refit.
    base = Genome.init(jr.PRNGKey(0), cfg)

    # ---- Stage 1: pure hand-wired oracle through the real dynamics ----------
    uf, neuron_to_row, hid_ids = base.unique_structure(cfg)
    w_full = (w_u_target[neuron_to_row][:, neuron_to_row]) * _role_mask(cfg)
    oracle_bias_genome = eqx.tree_at(lambda m: m.type_bias, base, type_bias)
    b_full = oracle_bias_genome.node_bias(cfg, hid_ids)
    out = run_dynamics(w_full, b_full, X_enc, cfg)
    preds = (out[:, 0] > 0).astype(jnp.int32)
    acc_oracle = balanced_accuracy(preds, y)
    print(f"[stage 1] hand-wired oracle (w,b) -> balanced accuracy {acc_oracle:.3f}")

    # ---- Stage 2: OLS-fit g to the oracle weights --------------------------
    g_fit, resid = fit_g_ols(base, cfg, w_u_target)
    print(f"[stage 2] OLS fit of g: max |w_fit - w_target| on allowed pairs = {resid:.4f}")

    # ---- Stage 3: build the oracle DNA and run the REAL forward pass --------
    oracle_dna = eqx.tree_at(lambda m: (m.g, m.type_bias), base, (g_fit, type_bias))
    preds_dna = jax.vmap(lambda o: oracle_dna.predict(o, cfg))(X_enc)[:, 0]
    acc_dna = balanced_accuracy(preds_dna, y)
    print(f"[stage 3] oracle DNA (fitted g) via real forward -> balanced accuracy {acc_dna:.3f}")

    print("\nverdict:")
    if acc_dna > 0.999:
        print(f"  perfect genome EXISTS for `{args.task}` (K={cfg.n_types}, g_width={cfg.g_width})")
        print("  -> at this capacity the failure is reachability/landscape, not representability.")
    elif acc_oracle > 0.999:
        print(f"  the IDEAL wiring solves `{args.task}` (stage 1) but the OLS-fitted g at width")
        print(f"  {cfg.g_width} cannot reproduce it (stage 3 {acc_dna:.3f}, residual {resid:.3f})")
        print("  -> g lacks capacity to express this wiring; representability is g-limited here.")
    else:
        print(f"  even the IDEAL hand-wired brain scores only {acc_oracle:.3f} at K={cfg.n_types}")
        print("  -> the perfect brain is NOT representable in this family (too few cell-types /")
        print("     wiring impossible). This is a representability wall, not a search failure.")


if __name__ == "__main__":
    main()
