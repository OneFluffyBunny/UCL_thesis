"""Quick smoke test for experiment_1.model. Not a real experiment -- just checks
that the genome builds a brain, inference runs, and per-type counts respond to
the abundance genes."""

import jax
import jax.numpy as jnp
import jax.random as jr

from model import BrainConfig, Genome, _role_mask as model_role_mask


def main():
    cfg = BrainConfig(n_in=8, n_hidden=20, n_out=1, n_types=4)
    key = jr.PRNGKey(0)
    g = Genome.init(key, cfg)

    # build the brain
    w, hid_ids = g.build_weights(cfg)
    assert w.shape == (cfg.n_total, cfg.n_total)

    # only IH / HH / HO edges should be non-zero
    n_in, n_hid = cfg.n_in, cfg.n_hidden
    assert jnp.all(w[:n_in, :n_in] == 0)                     # no input->input
    assert jnp.all(w[n_in + n_hid:, :] == 0)                # no output as source
    assert jnp.all(jnp.diag(w) == 0)                        # no self-loops
    assert jnp.all(jnp.abs(w) <= 1.0 + 1e-6)                # weights bounded [-1, 1]

    # g must be asymmetric -> directed graph (w != w.T on the HH block)
    hh = w[n_in:n_in + n_hid, n_in:n_in + n_hid]
    assert not bool(jnp.allclose(hh, hh.T)), "g is symmetric -> graph not directed"

    # optimised per-signature build must equal naive all-pairs build
    feats, _ = g.node_features(cfg)
    N = cfg.n_total
    fi = jnp.broadcast_to(feats[:, None, :], (N, N, cfg.feat_dim))
    fj = jnp.broadcast_to(feats[None, :, :], (N, N, cfg.feat_dim))
    w_naive = jax.vmap(jax.vmap(g.g))(jnp.concatenate([fi, fj], -1))[..., 0]
    w_naive = w_naive * model_role_mask(cfg)
    assert jnp.allclose(w, w_naive, atol=1e-6), "optimised build != naive all-pairs"

    # inference on a random batch of observations
    obs = jr.bernoulli(jr.PRNGKey(1), 0.5, (5, cfg.n_in)).astype(jnp.float32)
    preds = jax.vmap(lambda o: g.predict(o, cfg))(obs)
    assert preds.shape == (5, cfg.n_out)

    # counts: default (uniform abundance) ~ balanced; biased abundance shifts them
    counts_uniform = g.type_counts(cfg)
    g_biased = jax.tree_util.tree_map(lambda x: x, g)
    g_biased = type(g)(
        hidden_types=g.hidden_types, in_type=g.in_type, out_type=g.out_type,
        abundance=jnp.array([5.0, 0.0, -5.0, 0.0]), type_bias=g.type_bias, g=g.g,
    )
    counts_biased = g_biased.type_counts(cfg)

    print("w shape:            ", w.shape)
    print("nonzero edges:      ", int((w != 0).sum()))
    print("predictions:        ", preds.ravel().tolist())
    print("counts (uniform):   ", counts_uniform.tolist(), "sum", int(counts_uniform.sum()))
    print("counts (biased):    ", counts_biased.tolist(), "sum", int(counts_biased.sum()))
    assert int(counts_uniform.sum()) == cfg.n_hidden
    assert int(counts_biased.sum()) == cfg.n_hidden
    assert counts_biased[0] > counts_biased[2]              # type 0 favored, type 2 starved
    print("\nOK")


if __name__ == "__main__":
    main()
