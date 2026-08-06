import numpy as np
import psutil
import time
from utils import x0_sampling
import cma
import pandas as pd
from multiprocess import Pool


def CMAES(config, fitness, fitness_with_stats=None):
    nb_parameters = config["nb_trainable_parameters"]
    x0 = x0_sampling(config["x0_dist"], nb_parameters)
    es = cma.CMAEvolutionStrategy(
        x0,
        config["sigma_init"],
        {
            "verb_disp": config["print_every"],
            "popsize": config["popsize"],
            "maxiter": config["generations"],
            "seed": config["seed"],
            "CMA_elitist": config["CMA_elitist"],
            "minstd": config["minstd"],
        },
    )

    print("\n.......................................................")
    print("\nInitilisating CMA-ES with", nb_parameters, "trainable parameters \n")

    print("\n--- Starting Evolution ---\n")
    tic = time.time()
    gen_tic = time.time()

    objective_solution_best = np.inf
    objective_solution_centroid = np.inf
    objectives_centroid = []
    objectives_best = []
    gen = 0
    early_stopping_executed = False

    # Physical cores in the machine
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    print(f"\nUsing {num_cores} cores\n")

    # Baseline: fitness of the untrained x0 DNA (evolution's generation-zero starting point,
    # before any selection) -- reference point for "how good is the dumbest network".
    if fitness_with_stats is not None:
        baseline_fitval, baseline_raw, baseline_nodes = fitness_with_stats(x0)
        print(f"Baseline (untrained x0 DNA): fitness={baseline_fitval:.4f} (raw={baseline_raw:.1f}) | Nodes: {baseline_nodes}\n")
        config["baseline_x0_fitness"] = float(baseline_fitval)
        config["baseline_x0_raw"] = float(baseline_raw)
        config["baseline_x0_nodes"] = int(baseline_nodes)
    else:
        baseline_fitval = fitness(x0)
        print(f"Baseline (untrained x0 DNA): fitness={baseline_fitval:.4f}\n")
        config["baseline_x0_fitness"] = float(baseline_fitval)

    # Optimisation loop
    pool = Pool(num_cores) if num_cores > 1 else None
    while not es.stop() or gen < config["generations"]:
        try:
            # Generate candidate solutions
            X = es.ask()

            config["current_gen"] = gen
            warmup = config.get("size_reg_warmup")
            if warmup is not None and gen == warmup:
                print(f"\n  [Gen {gen}] Regularisation off — switching to raw fitness\n")

            # Evaluate in parallel — use stats variant to get raw reward + brain size
            use_stats = fitness_with_stats is not None
            eval_fn = fitness_with_stats if use_stats else fitness
            if pool is not None:
                results = pool.map(eval_fn, X)
            else:
                results = [eval_fn(x) for x in X]

            if use_stats:
                fitvals = [r[0] for r in results]
                raw_rewards = [r[1] for r in results]
                brain_sizes = [r[2] for r in results]
            else:
                fitvals = results

            # Correct sign — CMA-ES minimises, we maximise
            if config["maximise"]:
                fitvals_for_cma = [-f for f in fitvals]
            else:
                fitvals_for_cma = fitvals

            # Inform CMA optimizer of fitness results
            es.tell(X, fitvals_for_cma)

            if gen % config["print_every"] == 0:
                es.disp()
                pop_mean_score = np.mean(fitvals)

                extra = ""
                if use_stats:
                    best_idx = int(np.argmax(fitvals) if config["maximise"] else np.argmin(fitvals))
                    best_reg = fitvals[best_idx]
                    best_raw = raw_rewards[best_idx]
                    best_nodes = brain_sizes[best_idx]
                    mean_nodes = np.mean(brain_sizes)
                    extra += f" | Nodes: {best_nodes} (mean {mean_nodes:.1f})"
                    best_str = f"{best_reg:.2f} (raw: {best_raw:.1f})"
                else:
                    best_reg = max(fitvals) if config["maximise"] else min(fitvals)
                    best_str = f"{best_reg:.2f}"
                if config.get("log_gen_time", True):
                    elapsed = time.time() - gen_tic
                    extra += f" | {elapsed/config['print_every']:.1f}s/gen"
                    gen_tic = time.time()
                if config.get("size_reg_warmup") is not None:
                    extra += " | [reg]" if gen < config["size_reg_warmup"] else " | [raw]"
                print(f"  Gen {gen:4d} | Best: {best_str} | Pop mean: {pop_mean_score:.2f} | Sigma: {es.sigma:.4f}{extra}")

            # Store best solution
            objective_current_best_sol = es.best.f
            objectives_best.append(objective_current_best_sol)
            if objective_current_best_sol <= objective_solution_best:
                objective_solution_best = objective_current_best_sol
                solution_best = es.best.x

            # Store best mean solution
            objective_current_centroid_sol = np.mean(es.fit.fit)
            objectives_centroid.append(objective_current_centroid_sol)
            if objective_current_centroid_sol <= objective_solution_centroid:
                objective_solution_centroid = objective_current_centroid_sol
                solution_centroid = es.mean

            # Success stop: best fitness reached the target
            target = config.get("target")
            if target is not None and config["maximise"] and -objective_current_best_sol >= target:
                print(f"\nTarget fitness {target} reached at generation {gen}. Stopping.\n")
                break

            gen += 1

            if gen % config["evolution_feval_check_every"] == 0:
                test_fevals = []
                print("\n" + 30 * "v")
                for _ in range(config["evolution_feval_check_N"] // config["nb_episode_evals"]):
                    checksum = True if _ == 0 else False
                    # feval = fitness(solution_best, config, render=False, solution_id="best_check_"+str(config['evolution_feval_check_N']), checksum=checksum)
                    feval = fitness(solution_best)
                    test_fevals.append(feval)
                print(f"\nEvaluated {config['nb_episode_evals']*(config['evolution_feval_check_N']//config['nb_episode_evals'])} times the best solution found so far. Mean: {np.mean(test_fevals)}")
                print("\n" + 30 * "^" + "\n")

        # Allows to interrupt optimation with Ctrl+C
        except KeyboardInterrupt:  # Only works with python mp
            time.sleep(5)
            print("\n" + 20 * "*")
            print(f"\nCaught Ctrl+C!\nStopping evolution\n")
            print(20 * "*" + "\n")
            break

        # Early stopping of evolution
        if config["early_stopping"]:
            if gen == config["early_stopping_conditions"]["generation"] and objective_current_best_sol > config["early_stopping_conditions"]["objective_value"]:
                print(f"\nObjective too high {objective_current_best_sol} (reward too low) at generation {gen}.\nUnpromising run! Stopping evolution.\n")
                print(20 * "*")
                early_stopping_executed = True
                break

        # # Stopping evolution if loss flattening
        if config["flattening_stopping"] and gen > config["flattening_stopping_conditions"]["min_generation"]:
            std_objective = np.std(objectives_best[: -config["flattening_stopping_conditions"]["last_generations"]])
            if std_objective < config["flattening_stopping_conditions"]["min_std"]:
                print(f"\nObjective flattening!\nStd last {config['flattening_stopping_conditions']['last_generations']} generation is: {std_objective}\nStopping evolution.\n")
                print(20 * "*")
                break

    if pool is not None:
        pool.terminate()

    # losses/Loss arrays
    objectives_centroid = np.array(objectives_centroid)
    objectives_best = np.array(objectives_best)
    # solution = es.result # unsed since ask&tell

    toc = time.time()
    config["training time"] = str(int(toc - tic)) + " seconds"
    print("\nEvolution took: ", int(toc - tic), " seconds\n")
    print(f"========Optimizer output:==========================")
    print(f"Best single loss found was {objective_solution_best}")
    print(f"Best population centroid loss found was {objective_solution_centroid}")
    print(f"===================================================\n")

    # Create dataframe for logging objective values
    logger_pd = pd.DataFrame({"pop_best_eval": objectives_best, "mean_eval": objectives_centroid})

    return solution_best, solution_centroid, early_stopping_executed, logger_pd
