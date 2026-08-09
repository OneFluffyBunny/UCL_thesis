# Kashtan–Alon 2005 — paper specification (source of truth)

The specification of the **neural-network / retina experiment** (Fig. 5) of
Kashtan & Alon, *Spontaneous evolution of modularity and network motifs*, PNAS
102(39):13773–13778, doi:10.1073/pnas.0503610102. Full text fetched from
**PMC1236541** on **2026-08-04**.

This file is the paper, not our code. Every row is tagged:

- ✅ **QUOTED** — verbatim in the main text; the quote is given.
- 🔷 **NOT STATED** — absent from the main text (lives in the SI, or unstated);
  any value we use is a *reconstruction*, not the paper.
- ⚠️ **AMBIGUOUS** — mentioned but underspecified; our reading is one choice.

> ⚠️ This is the RETINA / NEURAL-NETWORK experiment. Kashtan–Alon's *other*
> experiment (electronic NAND circuits, the XOR-of-pairs logic task) has its own,
> DIFFERENT numbers (S=1000, L=300, Pm=0.7, Q_m 0.54 vs 0.12). **Never import a
> circuit number into the neural-net spec.** Where the paper gives a value only for
> the circuit experiment, it is marked as such below.

---

## 1. Network architecture

| item | status | paper |
|---|---|---|
| Layers of neurons | ✅ QUOTED | *"The neurons were set in four layers with eight, four, two, and one neuron per layer."* |
| Feed-forward, adjacent layers only | ✅ QUOTED | *"Connections were only between neighboring layers in a feed-forward manner."* |
| Retina feeds first layer only | ✅ QUOTED | *"Connections from the retina were to the first layer only."* → node graph = **retina(8) → 8 → 4 → 2 → 1** (8 pixels are a separate input layer). |
| Weights ±1 | ✅ QUOTED | *"Each connection had weight –1 or 1."* |
| **Parallel connections sum** | ✅ QUOTED | *"If two of the inputs were from the same neuron then the effective weight of the connection was the sum of the weights of the two connections."* → **multiple edges between the same neuron pair are allowed; effective weight can exceed ±1.** |
| Fan-in limits | ✅ QUOTED | *"Each neuron was given a maximal number of incoming connections as follows: three inputs for a neuron in the first, second, and third layer and two inputs for a neuron in the fourth layer."* → **(3,3,3,2)**. Counts *connections*, which (per the row above) may repeat a source. |
| Neuron activation rule / threshold / output values | 🔷 NOT STATED | The main text does **not** give the activation function, threshold, or output range. (Threshold units with {0,1} output is a standard reconstruction, but it is *not* quoted.) |
| Genome | ✅ QUOTED (count) | 15 neurons ⇒ 15 genes (one per neuron, encoding that neuron's incoming connections). |

## 2. The task (retina, Fig. 5a)

| item | status | paper |
|---|---|---|
| Retina size | ✅ QUOTED | 4 wide × 2 high = 8 pixels, left 2×2 block + right 2×2 block. |
| Left object | ✅ QUOTED | *"A left object is defined by three or more black pixels or one or two black pixels in the left column only."* |
| Right object | ✅ QUOTED | *"A right object is defined in a similar way, with one or two black pixels in the right column only."* |
| Two goals | ✅ QUOTED | *"L AND R, and L OR R"* (L = left object present, R = right object present). |
| Pixel → input-index mapping | 🔷 NOT STATED | Only shown as the Fig. 5a picture; we choose left = the 4 left-block pixels, right = the 4 right-block pixels. |

## 3. The MVG experiment

| item | status | paper |
|---|---|---|
| MVG vs Fixed-Goal | ✅ QUOTED | MVG alternates the two goals (shared L/R sub-goals, only the top AND/OR combiner changes); Fixed-Goal holds one goal. |
| Goal-switch epoch E | ⚠️ AMBIGUOUS | *"The goal was switched every E = 20 generations"* is stated for the **circuit** experiment; the fetch found it **NOT explicitly restated for the neural network**. We use E=20 for the retina as a reconstruction. |

## 4. The genetic algorithm (retina/NN)

| item | status | paper |
|---|---|---|
| Population S | ✅ QUOTED | **S = 600.** |
| Crossover probability | ✅ QUOTED | **P_c = 0.5**; *"a crossover was performed between two randomly chosen network genomes with a probability P_c."* |
| Mutation probability | ✅ QUOTED | **P_m = 0.5 per genome.** |
| Elite strategy present | ✅ QUOTED | An elite strategy is used. |
| **Elite count L** | 🔷 NOT STATED | The elite *count* for the NN is **not given** in the main text. Our L=150 is a reconstruction (the circuit experiment uses L=300 of S=1000). |
| Crossover parent pool | ⚠️ AMBIGUOUS | *"two randomly chosen network genomes"* — not stated whether drawn from the whole population or only the elite. Our code draws from the elite. |
| Crossover operator (how genomes mix) | 🔷 NOT STATED | Mechanism not in the main text (SI). Our per-destination-neuron column inheritance is a reconstruction. |
| Mutation operators (the set) | 🔷 NOT STATED | *"Mutations and crossovers were used as evolutionary operators"* — the operator set (add/remove edge, flip weight, change threshold) is **not** in the main text (SI). Reconstruction. |
| Generations to solution | ✅ QUOTED | MVG: *"within 2,800 (+9,500, –600) generations"*; Fixed-Goal: *"after 21,000 (+29,000, –3,600) generations."* |

## 5. Fitness

| item | status | paper |
|---|---|---|
| Measure | ✅ QUOTED | *"The fitness of a network was defined by the fraction of correct recognitions in this environment."* (raw fraction correct, no class balancing). |
| **Evaluation set** | ✅ QUOTED | *"The environment contained 100 different randomly chosen retina patterns."* → **fitness is over a 100-pattern sample, NOT all 256 patterns.** |
| Resampling frequency of the 100 patterns | ⚠️ AMBIGUOUS | Not stated whether the 100 patterns are fixed or redrawn each generation. |

## 6. Modularity measurement

| item | status | paper |
|---|---|---|
| Q (Newman) | ✅ QUOTED | *"the fraction of the edges in the network that connect between nodes in a module minus the expected value of the same quantity in a network with the same assignment of nodes into modules but random connections between the nodes."* Partition chosen by the **Newman–Girvan** algorithm (maximizes Q; module count not fixed). |
| Undirected graph for Q | ✅ QUOTED | *"we first converted the network into a nondirected graph by ignoring edge directionality and calculated its Q_real."* |
| Q_m formula | ✅ QUOTED | Q_m = (Q_real − Q_rand)/(Q_max − Q_rand); *"Q_max is defined as the maximal possible Q value of a network with the same degree sequence as the real network."* |
| Q_rand — control 1 | ✅ QUOTED | *"For the first control, we used randomized networks that preserve the degree sequence of the real network."* → **degree sequence only.** (No mention of layered structure *for Q_m*.) |
| Q_rand — control 2 | ✅ QUOTED | *"For the second control, we computed the Q of networks coded by random genomes that mapped to networks with the same number of nodes as in the real network, using the same genome definition and genotype–phenotype mapping as in the experiment."* → this control **does** respect the layered/fan-in structure (it goes through the real encoding). The two controls *"yielded similar results."* |
| Number of randomizations | ✅ QUOTED | *"We used 1,000 random networks for computing Q_rand."* |
| **Q_max method** | ✅ QUOTED | *"To estimate Q_max we repeated the evolution simulations, with exactly the same settings … where instead of evolving the networks toward the original information processing goal, we define the goal as maximizing the modularity measure Q. Q_max was defined as the average Q over 100 simulations of the best evolved network."* → **Q_max is obtained by RE-EVOLVING toward Q (100 sims), NOT by an edge-swap hill-climb.** |
| "Layered structure preserved" | ✅ QUOTED (different analysis) | This phrase appears only for **motif detection**, not Q_m: *"The randomized networks preserved the single-node characteristics of the real network, such as the incoming and outgoing degree sequence (and the layered structure in the case of neural networks…)."* Do **not** attribute it to the Q_m null. |

## 7. Results (neural network)

| condition | status | Q_m |
|---|---|---|
| Modularly varying goals | ✅ QUOTED | **Q_m = 0.35 ± 0.02** |
| Fixed goal | ✅ QUOTED | **Q_m = 0.15 ± 0.02** |
