import pathlib
import time
import yaml
import numpy as np
import warnings
import torch

from train_backend import train_model, grow_network, snapshot_graph_png, env_rollout, retina_fitness
from utils import seed_python_numpy_torch_cuda, visualise_graph, environment_max_reward

from tests_checks.test_config import all_config_checks

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)


def train(config):
    pathlib.Path(config["_path"]).mkdir(parents=True, exist_ok=False)

    ########
    # Seed #
    ########
    config["seed"] = np.random.randint(10**7) if config["seed"] is None else config["seed"]
    seed_python_numpy_torch_cuda(config["seed"])
    print("\nSeed: ", config["seed"])
    print("Env_seed: ", config["env_seed"])

    ########################
    ### Launch traininig ###
    ########################
    solution_best, solution_centroid, early_stopping_executed, logger_df = train_model(config)

    if not early_stopping_executed:
        # Save results
        print(f"\nSaving models and config file — Run ID {config['id']}")
        print(f"\nFinal model has {config['nb_trainable_parameters']} parameters")
        if config["save_model"]:
            # Save solution and rewards
            np.save(path + "/" + "solution_centroid", solution_centroid)
            np.save(path + "/" + "solution_best", solution_best)
            if logger_df is not None:
                logger_df.to_csv(path + "/" + "logger.csv")

        # Save config file
        with open(config["_path"] + "/" + "config.yml", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # Static graph snapshot (no ffmpeg required)
        if config.get("snapshot"):
            png_path = config["_path"] + "/graph_best.png"
            W, network_state = grow_network(solution_best, config)

            extra_title = f"{len(solution_best)} genes"
            if "retina" in config["environment"]:
                balanced = config.get("balanced_fitness", False)
                max_reward = environment_max_reward(config["environment"], balanced=balanced)
                score = retina_fitness(W=W, config=config)
                if balanced:
                    extra_title += f"\nBalanced accuracy: {score:.4f} ({100 * score:.1f}%)"
                else:
                    extra_title += f"\nAccuracy: {score} / {max_reward} ({100 * score / max_reward:.1f}%)"
            elif "Network" not in config["environment"] and "gate" not in config["environment"]:
                try:
                    max_reward = environment_max_reward(config["environment"])
                    n_eval = config["nb_eval_seeds"]
                    scores = [env_rollout(W=W, config=config, seed=i) for i in range(n_eval)]
                    avg_score = sum(scores) / len(scores)
                    extra_title += f"\nAvg over {n_eval} seeds: {avg_score:.1f} / {max_reward} ({100 * avg_score / max_reward:.1f}%)"
                except NotImplementedError:
                    pass

            snapshot_graph_png(W, network_state, config, png_path, extra_title=extra_title)

            # Growth-stages grid (seed -> final brain), same DNA, only if node_based_growth
            stages_path = None
            if not config["node_pairs_based_growth"]:
                from growth_stages import grow_with_stages, render_growth_stages_png

                stages_path = config["_path"] + "/growth_stages.png"
                stages = grow_with_stages(solution_best, config)
                render_growth_stages_png(stages, config, stages_path, run_id=config["id"])

            if config.get("show"):
                import os
                os.startfile(os.path.abspath(png_path))
                if stages_path is not None:
                    os.startfile(os.path.abspath(stages_path))

        # Visaulise graph development
        if config["visualise_network"]:
            config["nb_episode_evals"] = 1
            config["nb_growth_evals"] = 1
            print(f"\nGenerating graph visualisations...")
            pathlib.Path(config["_path"] + "/graph_animations/").mkdir(parents=True, exist_ok=False)
            print(f"Generating growth visualisations")
            visualise_graph(solution_best, config, "Graph development — Best solution", env_rollout=False)
            visualise_graph(solution_centroid, config, "Graph development — Centroid solution", env_rollout=False)
            if "Network" not in config["environment"]:
                print(f"Generating rollout visualisations")
                visualise_graph(solution_best, config, "Information propagation during rollout — Best solution", env_rollout=True)
                visualise_graph(solution_centroid, config, "Information propagation during rollout — Centroid solution", env_rollout=True)

        # Render
        if config["render"]:
            from train_backend import fitness_functional

            pathlib.Path(config["_path"] + "/env_renders/").mkdir(parents=True, exist_ok=False)
            config["nb_episode_evals"] = 1
            config["nb_growth_evals"] = 3
            print(f"\nRunning environment for best solution...")
            fitness = fitness_functional(config, render=True, solution_id="best")
            fitness(solution_best)
            print(f"\nRunning environment for centroid solution...")
            print(f"\nRunning environment for centroid solution...")
            fitness = fitness_functional(config, render=True, solution_id="centroid")

        print(f"\nBYE! - ID: {config['id']}")

    else:
        print(f"\nEARLY STOPPING EXECUTED\nNothing will remain, bye!\n")


if __name__ == "__main__":
    # Load configuration file
    import argparse

    parser = argparse.ArgumentParser(description="Configuration file path")
    parser.add_argument("--conf", type=str, default="run_experiment.yaml", help="Path to yaml configuration file")
    parser.add_argument("--profile", action="store_true", help="Print per-component timing breakdown after each fitness eval (implies --threads 1)")
    parser.add_argument("--generations", type=int, default=None, help="Override generations from config")
    parser.add_argument("--threads", type=int, default=None, help="Override threads from config")
    parser.add_argument("--popsize", type=int, default=None, help="Override popsize from config")
    parser.add_argument("--visualise", action="store_true", help="Enable graph visualisation (off by default, requires ffmpeg)")
    parser.add_argument("--save-dna", action="store_true", help="Save best and centroid DNA as .npy files after training")
    parser.add_argument("--snapshot", action="store_true", help="Save a static PNG of the final grown graph for the best solution")
    parser.add_argument("--show", action="store_true", help="Save and open the best brain PNG after training (implies --snapshot)")
    parser.add_argument("--nb-episode-evals", type=int, default=None, help="Override nb_episode_evals from config")
    parser.add_argument("--sigma-init", type=float, default=None, help="Override sigma_init from config")
    parser.add_argument("--growth-cycles", type=int, default=None, help="Override number_of_growth_cycles from config")
    parser.add_argument("--no-gen-time", action="store_true", help="Disable per-generation timing in progress output")
    parser.add_argument("--pruning", action="store_true", default=False, help="Enable edge pruning after each growth cycle (overrides config)")
    parser.add_argument("--no-elitism", action="store_true", default=False, help="Disable CMA-ES elitism (default is elitist)")
    parser.add_argument("--growth-threshold", type=float, default=None, help="Growth MLP output must exceed this to spawn a node (default 0.0)")
    parser.add_argument("--size-reg", type=str, default=None, help="Brain size regularisation strategy: 'io_ratio' (nodes), 'io_edges' (edges), 'both' (nodes + edges)")
    parser.add_argument("--size-reg-alpha", type=float, default=None, help="Node regularisation strength (default 1.0)")
    parser.add_argument("--size-reg-alpha-edges", type=float, default=None, help="Edge regularisation strength (default 1.0)")
    parser.add_argument("--size-reg-warmup", type=int, default=None, help="Apply size regularisation for the first N generations only; switches to raw fitness afterwards")
    parser.add_argument("--target", type=float, default=None, help="Stop evolution as soon as best fitness reaches this value (default: env max reward)")
    parser.add_argument("--nb-eval-seeds", type=int, default=10, help="Number of seeds to average the best brain's score over for the snapshot title (default 10)")
    parser.add_argument("--balanced-fitness", action="store_true", default=False, help="Retina task: use balanced accuracy (mean of per-class accuracy, in [0,1]) instead of raw correct-count as the fitness signal (default False)")
    parser.add_argument("--no-early-stopping", action="store_true", default=False, help="Disable the 'unpromising run' early-stopping check (early_stopping_conditions). Its objective_value threshold is tuned for raw-reward scales (e.g. -3) and will always trigger on a [0,1] balanced-fitness run, silently discarding logs/snapshot -- pass this flag for any --balanced-fitness run")
    parser.add_argument("--allow-io-self-edges", action="store_true", default=False, help="Allow input-input and output-output edges in the seed graph (forbidden by default -- an input node is clamped to the observation every propagation step during rollout, so an edge into it from another input node, or itself, can never affect anything)")
    args = parser.parse_args()
    with open(args.conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if args.profile:
        config["profile"] = True
        if args.threads is None:
            config["threads"] = 1
    if args.generations is not None:
        config["generations"] = args.generations
    if args.threads is not None:
        config["threads"] = args.threads
    if args.popsize is not None:
        config["popsize"] = args.popsize
    if args.nb_episode_evals is not None:
        config["nb_episode_evals"] = args.nb_episode_evals
        n = args.nb_episode_evals
        feval_n = config["evolution_feval_check_N"]
        config["evolution_feval_check_N"] = max(n, (feval_n // n) * n)
    if args.sigma_init is not None:
        config["sigma_init"] = args.sigma_init
    if args.growth_cycles is not None:
        config["number_of_growth_cycles"] = args.growth_cycles
    config["log_gen_time"] = not args.no_gen_time
    if args.pruning:
        config["prunning_phase"] = True
    if args.no_elitism:
        config["CMA_elitist"] = False
    if args.growth_threshold is not None:
        config["growth_threshold"] = args.growth_threshold
    if args.size_reg is not None:
        config["size_regularisation"] = args.size_reg
    if args.size_reg_alpha is not None:
        config["size_reg_alpha"] = args.size_reg_alpha
    if args.size_reg_alpha_edges is not None:
        config["size_reg_alpha_edges"] = args.size_reg_alpha_edges
    config["visualise_network"] = 1 if args.visualise else 0
    if args.save_dna:
        config["save_model"] = True
    config["snapshot"] = args.snapshot or args.show
    config["show"] = args.show
    config["nb_eval_seeds"] = args.nb_eval_seeds
    config["balanced_fitness"] = args.balanced_fitness
    if args.no_early_stopping:
        config["early_stopping"] = False
    if args.allow_io_self_edges:
        config["forbid_io_self_edges"] = False
    if args.size_reg_warmup is not None:
        config["size_reg_warmup"] = args.size_reg_warmup
    if args.target is not None:
        config["target"] = args.target
    else:
        from utils import environment_max_reward
        try:
            config["target"] = environment_max_reward(config["environment"], balanced=config["balanced_fitness"])
        except NotImplementedError:
            config["target"] = None

    # Check config file makes sense
    all_config_checks(config)

    # Create local path to save results locally if not resuming training from checkpoint
    config["id"] = str(int(time.time()))
    print(f"Model ID: {config['id']}")
    path = "saved_models" + "/" + str(config["id"])
    config["_path"] = path

    print(f"\nConfig:\n{config}")

    # Launch training
    train(config)
