import pathlib
import time
import yaml
import numpy as np
import warnings
import torch

from train_backend import train_model, grow_network, snapshot_graph_png
from utils import seed_python_numpy_torch_cuda, visualise_graph

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
            snapshot_graph_png(W, network_state, config, png_path)
            if config.get("show"):
                import os
                os.startfile(os.path.abspath(png_path))

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
    parser.add_argument("--size-reg", type=str, default=None, help="Brain size regularisation strategy. Options: 'io_ratio' (penalise hidden nodes relative to seed size)")
    parser.add_argument("--size-reg-alpha", type=float, default=None, help="Strength of size regularisation penalty (default 1.0)")
    parser.add_argument("--target", type=float, default=None, help="Stop evolution as soon as best fitness reaches this value (default: env max reward)")
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
    if args.size_reg is not None:
        config["size_regularisation"] = args.size_reg
    if args.size_reg_alpha is not None:
        config["size_reg_alpha"] = args.size_reg_alpha
    config["visualise_network"] = 1 if args.visualise else 0
    if args.save_dna:
        config["save_model"] = True
    config["snapshot"] = args.snapshot or args.show
    config["show"] = args.show
    if args.target is not None:
        config["target"] = args.target
    else:
        from utils import environment_max_reward
        try:
            config["target"] = environment_max_reward(config["environment"])
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
