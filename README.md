# Representation-Guided Reinforcement Learning for Guideline-Compliant Multi-Agent Orchestration

This repository contains the official, production-ready implementation of the **VHAS (Vietnam Health-Agent System)**, a testbed designed to evaluate the **Representation-Guided Reinforcement Learning (RGRL)** framework. 

The primary objective of this work is to address the combinatorial explosion of state and action spaces in multi-agent reinforcement learning (RL) when operating in environments characterized by high-dimensional, unstructured text inputs (Semantic Complexity). The proposed system enforces strict guideline compliance via a Neuro-Symbolic Dual-System reasoning architecture.

---

## 1. System Architecture

The RGRL framework is implemented in two offline-online stages, integrated with an uncertainty-aware dual-system inference router at runtime:

### Stage 1: Workflow Representation Learning (WRL) [Offline]
- **Methodology:** A Dual-Encoder architecture (StateEncoder \& ActionEncoder) optimized via self-supervised Contrastive Learning (Multiple Negatives Ranking Loss).
- **Function:** Maps high-dimensional textual contexts (Semantic States) and static entity specifications (Actions) into a unified, continuous embedding space.
- **Output:** A static, pre-computed embedding matrix serving as a structured **Knowledge Map**.

### Stage 2: Guided Reinforcement Optimization (GRO) [Online]
- **Methodology:** Proximal Policy Optimization (PPO) combined with dynamic, representation-guided Action Masking.
- **Function:** At each decision step, the orchestrator queries the Knowledge Map to retrieve the $k$-nearest valid actions, generating a binary mask at the logit level. This strictly constrains RL exploration to compliant actions, preventing standard RL from suffering under sparse rewards.

### Inference: The Neuro-Symbolic Dual-System Router
To safeguard against model uncertainty in out-of-distribution (OOD) or highly ambiguous contexts, the inference layer utilizes a dynamic delegation strategy based on a confidence threshold ($\tau$):
- **System 1 (AFAN - Graph Neural Network):** A structural policy that models workflow history as a Heterogeneous Directed Graph using `GATv2Conv` layers. It executes fast, intuitive decisions when its confidence $C_t \ge \tau$.
- **System 2 (Deliberative Pipeline / Compliance Guardrail):** Activated as a safety fallback when $C_t < \tau$. It runs an iterative reasoning pipeline utilizing a 3-agent loop (**Generator $\rightarrow$ Verifier $\rightarrow$ Reviser**) to enforce hard constraints and correct structural deviations before final execution.
- **Human-in-the-Loop (HITL):** Suspends execution and escalates to a human domain expert if System 2 fails to reach a verified compliant consensus within a bounded number of retries ($N_{max}$).

---

## 2. Repository Structure

```text
rgrl/
├── ai/
│   ├── data/                           # Dataset split into 80/10/10 (1600/200/200 traces)
│   │   ├── clinical_notes/             # Filtered MIMIC-IV notes
│   │   ├── clinical_states/            # 35 Semantic Prototypes configuration
│   │   ├── scenarios/                  # Context scenarios JSON
│   │   ├── vhas_universe/              # Agent/Tool declarations
│   │   ├── workflow_skeletons/         # The 3 Ground-Truth Pathways
│   │   └── workflow_traces/            # Raw traces and splits
│   │       ├── train_traces.json       # 1,600 training samples
│   │       ├── val_traces.json         # 200 validation samples
│   │       ├── test_traces.json        # 200 held-out evaluation samples
│   │       └── prepare_splits.py       # Data stratification script
│   │
│   ├── dual_system/                    # Inference and Routing implementation
│   │   ├── neuro_symbolic_router.py    # GNN confidence-based routing wrapper
│   │   ├── deliberative_llm.py         # Generator-Verifier-Reviser pipeline
│   │   └── prompts.py                  # Strict operational guideline definitions
│   │
│   ├── experiments/                    # Official evaluation scripts for thesis metrics
│   │   ├── evaluate_tradeoff.py        # Pareto analysis over tau thresholds
│   │   └── evaluate_ablation.py        # System 2 structural ablation study
│   │
│   ├── gro/                            # Stage 2: Reinforcement Learning modules
│   │   ├── simulation/                 # Flat vector environment (GuidelineCompliantEnv)
│   │   ├── gnn/                        # AFAN GNN architecture and PyG env/train
│   │   ├── transformer/                # Sequential policy and sequential env
│   │   ├── train_mlp.py                # MLP policy training script (SB3)
│   │   ├── train_model_a_deployed.py   # Model A training deployment (Modal)
│   │   ├── train_model_b_deployed.py   # Model B training deployment (Modal)
│   │   ├── train_no_guidance_deployed.py # Vanilla RL training deployment (Modal)
│   │   └── train_topk_ablation_deployed.py # Top-k evaluation deployment (Modal)
│   │
│   ├── wrl/                            # Stage 1: Representation Learning modules
│   │   ├── adp/                        # General action sequences corpus
│   │   ├── pretraining/                # Encoder pre-training implementation (Modal)
│   │   ├── finetuning/                 # Encoder fine-tuning implementation (Modal)
│   │   │   └── script/
│   │   │       └── run_finetuning_jobs_dual.py # Standard dual fine-tuner
│   │   └── embedding_space/            # Logic for building and evaluating WRL maps
│   │       └── script/
│   │           ├── build_embedding_space.py  # Standard map builder
│   │           ├── evaluate_encoder.py       # Nearest-neighbor evaluator
│   │           └── visualize_embeddings.py   # t-SNE visualizer
│   │
│   ├── results/                        # Empirical result logs and visualizations
│   │   ├── figures/                    # Generated high-resolution plots used in the thesis
│   │   ├── training_dynamics.csv       # Per-timestep logs of RL convergence
│   │   ├── test_eval_200_episodes.csv  # Raw success rate data across 200 test cases
│   │   ├── pareto_tradeoff_results.csv # Results of the Safety vs. Efficiency sweep
│   │   ├── system2_ablation_results.csv # Metrics for the 3-agent pipeline comparison
│   │   ├── tsne_embeddings_2d.csv      # Coordinates for WRL t-SNE plot
│   │   └── plot_training_results.py    # Script to programmatically regenerate figures
│   │
│   ├── models/                         # Trained model artifacts and weights
│   │   ├── stage1_wrl_encoders/        # Fine-tuned State and Action encoders
│   │   └── stage2_gro_policies/        # SB3 (.zip) and RSL-RL/PyG (.pt) checkpoints
│   │
│   └── rsl_rl/                         # Customized RSL-RL framework supporting TensorDict
│
├── web/                                # Next.js Progressive Disclosure Frontend
│   └── backend/                        # FastAPI TraceStore service
│       ├── api.py                      # REST endpoints for trace and feedback queries
│       ├── models.py                   # SQLAlchemy database schemas
│       ├── store.py                    # VHASTraceStore database driver
│       └── seed_tracestore.py          # Script to populate the PostgreSQL database
│
├── docker-compose-vhas-db.yml          # Docker Compose configuration for PostgreSQL
└── .env.local                          # Local environment secrets (API Keys, DB URLs)
```

---

## 3. Experimental Setup and Bounded Scope

To systematically evaluate the algorithm, the VHAS testbed is strictly bounded to internal medicine triage workflows.
- **Action Space:** 5 specialized agents, 6 tools.
- **State Space Abstraction:** 35 Semantic State Prototypes, designed via a $7 \times 5$ task decomposition matrix to ensure RL tractability.
- **Ground-Truth Guidelines:** 3 deterministic workflow skeletons derived from established process mining logs (MIMIC-EL) and clinical protocols (ESI v4).

---

## 4. Evaluation Metrics

We quantify the orchestrator's performance using three strict, compliance-oriented metrics:
1.  **Guideline Compliance Rate (GCR \%):** Frequency of perfect adherence to the ground-truth workflow skeletons.
2.  **Protocol Redundancy Rate (PRR \%):** Frequency of unnecessary or repetitive agent actions.
3.  **Average Orchestration Steps:** Measured decision efficiency in finding the optimal path compared to a 6.0-step static baseline.

---

## 5. Usage \& Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (with CUDA support for PyTorch Geometric)
- PostgreSQL (or Docker)
- OpenAI API Key (configured in `.env` for System 2 evaluation)

### Installation
```bash
git clone https://github.com/quyk67uet/rgrl.git
cd rgrl
pip install -r web/backend/requirements.txt
```

### Prepare Dataset Splits
To partition the 2,000 synthetic traces into the standard 80/10/10 split (1,600 Train / 200 Val / 200 Test) with seed 42, execute:
```bash
python ai/data/workflow_traces/prepare_splits.py
```

### Running Evaluation Experiments
To execute the actual evaluation logic over the held-out Test Set (200 cases) using the real inference pipelines:

**1. Execute Pareto Trade-off Analysis ($\tau$ Sweep):**
```bash
python ai/experiments/evaluate_tradeoff.py
```
*This script iterates through $\tau$ values and records real-time routing decisions to `ai/results/pareto_tradeoff_results.csv`.*

**2. Execute System 2 Ablation Study:**
```bash
python ai/experiments/evaluate_ablation.py
```
*This script compares policy configurations and records metrics to `ai/results/system2_ablation_results.csv`.*

### Regenerating Publication Figures
To programmatically regenerate the training dynamics plot (`learning_curves_2.png`) and success rate distribution (`guideline_compliance_boxplot.png`) directly from the raw evaluation logs:
```bash
python ai/results/plot_training_results.py
```
*The newly plotted figures will be saved directly into the `ai/results/figures/` folder.*

### Running the Web Demo Database (Optional)
The Next.js UI connects to a PostgreSQL database managed by the FastAPI backend.

**1. Start the Database Container:**
```bash
docker-compose -f docker-compose-vhas-db.yml up -d
```

**2. Seed the Database:**
```bash
python web/backend/seed_tracestore.py
```

**3. Run the FastAPI Service:**
```bash
python web/backend/api.py
```
*The REST documentation will be accessible at `http://localhost:8001/docs`.*