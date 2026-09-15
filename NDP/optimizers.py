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

            # MVG: which goal is active this generation. Fixed under FG (config["mvg"]
            # falsy), alternates every mvg_switch_interval generations under --mvg.
            # gen is 0-indexed here, matching kashtan_alon/train.py::goal_op exactly.
            if config.get("mvg", False):
                mvg_ops = config.get("mvg_ops", ["and", "or"])
                switch_interval = config.get("mvg_switch_interval", 20)
                config["current_op"] = mvg_ops[(gen // switch_interval) % len(mvg_ops)]
            else:
                config["current_op"] = config.get("operation", "and")

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

            # This generation's champion (not best-ever) -- computed every generation
            # (not just at print_every) so it's accurate to the true final generation
            # when the loop ends. Needed because under MVG, "best fitness ever" isn't
            # meaningfully comparable across goal switches (a peak hit while the goal
            # was AND doesn't mean the same thing once the goal is OR) -- mirrors
            # kashtan_alon/train.py's final_indiv/final_fit/final_op, which is what
            # that reference actually saves/reports/visualises, not its own best-ever
            # tracker, for the same reason.
            best_idx = int(np.argmax(fitvals) if config["maximise"] else np.argmin(fitvals))
            final_gen_solution = X[best_idx]
            final_gen_fitness = fitvals[best_idx]
            final_gen_op = config.get("current_op")
            if use_stats:
                final_gen_raw = raw_rewards[best_idx]
                final_gen_nodes = brain_sizes[best_idx]

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
                    best_reg = final_gen_fitness
                    best_raw = final_gen_raw
                    best_nodes = final_gen_nodes
                    mean_nodes = np.mean(brain_sizes)
                    extra += f" | Nodes: {best_nodes} (mean {mean_nodes:.1f})"
                    best_str = f"{best_reg:.2f} (raw: {best_raw:.1f})"
                else:
                    best_reg = final_gen_fitness
                    best_str = f"{best_reg:.2f}"
                if config.get("log_gen_time", True):
                    elapsed = time.time() - gen_tic
                    extra += f" | {elapsed/config['print_every']:.1f}s/gen"
                    gen_tic = time.time()
                if config.get("size_reg_warmup") is not None:
                    extra += " | [reg]" if gen < config["size_reg_warmup"] else " | [raw]"
                if config.get("mvg", False):
                    extra += f" | op={final_gen_op}"
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

            # Success stop: best fitness reached the target. Disabled under MVG (matches
            # kashtan_alon/train.py's `if args.early_stop and (not args.mvg) and ...`) --
            # trivially satisfying one goal and stopping would mean the other goal was
            # never addressed, which defeats the point of alternating them.
            target = config.get("target")
            if target is not None and config["maximise"] and not config.get("mvg", False) and -objective_current_best_sol >= target:
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
    print(f"Best single loss found (peak, any goal) was {objective_solution_best}")
    print(f"Best population centroid loss found was {objective_solution_centroid}")

    # Under MVG, "best fitness ever" isn't comparable across goal switches (see the
    # comment above final_gen_solution), so the network actually returned/saved/
    # snapshotted is the LAST generation's champion instead -- matching
    # kashtan_alon/train.py, which reports both (peak fitness, for "did it ever solve
    # either goal") but saves/visualises the final-generation network. FG runs are
    # completely unaffected (config["mvg"] is falsy by default): solution_best/
    # solution_centroid stay exactly the best-ever values, as before this feature.
    if config.get("mvg", False):
        peak_fitness = -objective_solution_best if config["maximise"] else objective_solution_best
        print(f"Final-generation champion (op={final_gen_op}): fitness={final_gen_fitness:.4f} "
              f"-- this is what gets returned/saved as solution_best under MVG "
              f"(peak fitness {peak_fitness:.4f} was reached under some possibly-different goal)")
        config["mvg_peak_fitness"] = float(peak_fitness)
        config["mvg_final_gen_fitness"] = float(final_gen_fitness)
        config["mvg_final_gen_op"] = final_gen_op
        solution_best = final_gen_solution
        solution_centroid = es.mean
    print(f"===================================================\n")

    # Create dataframe for logging objective values
    logger_pd = pd.DataFrame({"pop_best_eval": objectives_best, "mean_eval": objectives_centroid})

    return solution_best, solution_centroid, early_stopping_executed, logger_pd
