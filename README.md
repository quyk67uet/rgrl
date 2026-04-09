# Representation-Guided Reinforcement Learning for Guideline-Compliant Multi-Agent Orchestration under Semantic Variability

This repository contains the official implementation of the **VHAS (Vietnam Health-Agent System)**, a testbed designed to evaluate the **Representation-Guided Reinforcement Learning (RGRL)** framework. 

The primary objective of this work is to address the combinatorial explosion of state and action spaces in multi-agent reinforcement learning (RL) when operating in environments characterized by high-dimensional, unstructured text inputs (Semantic Variability). The proposed system enforces strict adherence to predefined operational protocols (Guideline Compliance) via a Neuro-Symbolic Dual-System architecture.

## 1. System Architecture

The RGRL framework is implemented in two main stages, integrating a dual-system inference routing mechanism during deployment:

### Stage 1: Workflow Representation Learning (WRL)
- **Methodology:** A Dual-Encoder architecture optimized via self-supervised Contrastive Learning (Multiple Negatives Ranking Loss).
- **Function:** Maps highly variable textual contexts (Semantic States) and static entity specifications (Agents/Tools) into a unified, continuous embedding space, forming a static Knowledge Map.

### Stage 2: Guided Reinforcement Optimization (GRO)
- **Methodology:** Proximal Policy Optimization (PPO) combined with dynamic Action Masking.
- **Function:** At each timestep, the system queries the Knowledge Map to retrieve top-$k$ relevant entities, generating a binary mask. This masks invalid logits before the Softmax layer, pruning the action space and stabilizing RL convergence.

### The Neuro-Symbolic Dual-System Router (Inference)
To handle boundary cases and model uncertainty, the inference layer utilizes a dynamic delegation strategy based on a confidence threshold ($\tau$):
- **System 1 (AFAN - Graph Neural Network):** A structural policy that models workflow history as a Heterogeneous Directed Graph using `GATv2Conv` layers. Provides high-speed, intuitive action predictions. Executed if $U_t \ge \tau$.
- **System 2 (Deliberative LLM Pipeline):** Activated when $U_t < \tau$. An iterative reasoning pipeline utilizing a 3-agent loop (**Generator $\rightarrow$ Verifier $\rightarrow$ Reviser**) to enforce hard constraints and correct structural deviations before final execution.

## 2. Repository Structure

The codebase is modularized to separate core algorithms, evaluation logic, and reproducibility scripts.

```text
THESIS_RGRL/
├── ai/
│   ├── data/                   # Synthetic benchmark dataset (2,000 traces)
│   │   ├── scenarios/          # Extracted clinical contexts
│   │   └── workflow_traces/    # Raw traces + ML splits (train/val/test)
│   │
│   ├── wrl/                    # Stage 1 implementation
│   │   ├── pretraining/        # Scripts for ADP/T1 corpus pretraining
│   │   └── finetuning/         # Scripts for contrastive learning on domain traces
│   │
│   ├── gro/                    # Stage 2 implementation
│   │   ├── simulation/         # Custom Gymnasium environment with rule-based transitions
│   │   └── scripts/            # PPO training loops for MLP, Transformer, and GNN
│   │
│   ├── dual_system/            # Inference and Routing implementation
│   │   ├── neuro_symbolic_router.py # Threshold-based delegation logic
│   │   ├── deliberative_llm.py      # Generator-Verifier-Reviser pipeline
│   │   └── prompts.py               # Strict operational guideline definitions
│   │
│   ├── experiments/            # Real evaluation logic for thesis metrics
│   │   ├── evaluate_tradeoff.py     # Pareto analysis over tau thresholds
│   │   └── evaluate_ablation.py     # System 2 structural ablation study
│   │
│   ├── results/                # Empirical result logs and visualizations
│   │   ├── figures/                 # Generated high-resolution plots used in the paper
│   │   ├── training_dynamics.csv    # Per-timestep logs of RL convergence
│   │   ├── test_eval_200_episodes.csv # Raw success rate data across 200 test cases
│   │   ├── pareto_tradeoff_results.csv # Results of the Efficiency vs. Safety sweep
│   │   ├── system2_ablation_results.csv # Metrics for the 3-agent pipeline comparison
│   │   └── tsne_embeddings_2d.csv   # 2D coordinates for embedding space visualization
│   │
│   ├── models/                 # Checkpoints and frozen model weights
│   │   ├── stage1_wrl_encoders/     # SentenceTransformer configurations
│   │   └── stage2_gro_policies/     # SB3 (.zip) and RSL-RL/PyG (.pt) checkpoints
│   │
│   └── rsl_rl/                 # Customized RSL-RL framework supporting TensorDict
│
└── web/                        # Next.js Frontend for Progressive Disclosure Demo
	└── backend/                # FastAPI TraceStore service and database models
```

## 3. Experimental Setup and Bounded Scope

To systematically evaluate the algorithm, the VHAS testbed is strictly bounded to internal medicine triage workflows.
- **Action Space:** 5 specialized agents (`TriageAgent`, `EHRAgent`, `DispensationAgent`, `ReconciliationAgent`, `SummaryAgent`) and 6 functional tools.
- **State Space Abstraction:** 35 Semantic State Prototypes, designed via a $7 \times 5$ task decomposition matrix to ensure RL tractability while representing core clinical handoffs.
- **Ground-Truth Guidelines:** 3 deterministic workflow skeletons derived from established process mining logs (MIMIC-EL) and clinical protocols (ESI v4).

The evaluation dataset consists of 2,000 semi-synthetic traces acting as "semantic noise" to stress-test the algorithm's capability to adhere to the 3 ground-truth skeletons.

## 4. Evaluation Metrics

System performance is quantified using metrics designed to evaluate orchestration reliability:
1. **Guideline Compliance Rate (GCR \%):** The frequency at which the agent perfectly adheres to the prescribed workflow skeleton without deviation.
2. **Protocol Redundancy Rate (PRR \%):** The frequency of unnecessary or looping actions executed by the agent.
3. **Average Orchestration Steps:** Evaluates the efficiency of the policy in finding the shortest valid path compared to a static, rule-based baseline.

## 5. Getting Started & Reproducibility

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (with CUDA support for PyTorch Geometric)
- OpenAI API Key (for System 2 evaluation)

### Installation
```bash
git clone <repository_url>
cd THESIS_RGRL
pip install -r web/backend/requirements.txt
```

### Prepare Train/Val/Test Splits
The thesis dataset split is deterministic with seed 42:
- Train: 1,600
- Validation: 200
- Test: 200

```bash
python ai/data/workflow_traces/prepare_splits.py
```

### Running Experiments
The `ai/experiments/` directory contains the core logic for testing the architecture against the held-out Test Set (200 cases).

**1. Run Pareto Trade-off Analysis (Efficiency vs. Safety):**
```bash
python ai/experiments/evaluate_tradeoff.py
```

**2. Run System 2 Ablation Study:**
```bash
python ai/experiments/evaluate_ablation.py
```

### Running TraceStore Backend (Optional)
The TraceStore API for workflow browsing and feedback is implemented in `web/backend`.

```bash
python web/backend/api.py
```

API docs:
- `http://localhost:8001/docs`
- `http://localhost:8001/redoc`

### Seeding TraceStore Database (Optional)
Before seeding, make sure split files exist and PostgreSQL is running.

```bash
python ai/data/workflow_traces/prepare_splits.py
python web/backend/seed_tracestore.py
```

---