# Representation-Guided Reinforcement Learning for Guideline-Compliant Multi-Agent Orchestration

This repository contains the official, production-ready implementation of the **VHAS (Vietnam Health-Agent System)**, a testbed designed to evaluate the **Representation-Guided Reinforcement Learning (RGRL)** framework. 

The primary objective of this work is to address the combinatorial explosion of state and action spaces in multi-agent reinforcement learning (RL) when operating in environments characterized by high-dimensional, unstructured text inputs. The proposed system enforces strict guideline compliance via an uncertainty-aware, Neuro-Symbolic Dual-System reasoning architecture.

---

## 1. System Architecture

The RGRL framework is implemented in two offline-online stages, integrated with a dual-system inference routing mechanism during deployment:

### Stage 1: Workflow Representation Learning (WRL) [Offline]
- **Methodology:** A Dual-Encoder architecture (StateEncoder \& ActionEncoder) optimized via self-supervised Contrastive Learning (Multiple Negatives Ranking Loss).
- **Function:** Maps highly variable textual contexts (Semantic States) and static entity specifications (Actions) into a unified, continuous embedding space, forming a static Knowledge Map.

### Stage 2: Guided Reinforcement Optimization (GRO) [Online]
- **Methodology:** Proximal Policy Optimization (PPO) combined with dynamic, representation-guided Action Masking.
- **Function:** At each decision step, the orchestrator queries the Knowledge Map to retrieve the $k$-nearest valid actions, generating a binary mask at the logit level. This strictly constrains RL exploration to compliant actions, preventing standard RL from suffering under sparse rewards.

### System 1: The Agent-Flow Attention Network (AFAN) & RSL-RL Integration
System 1 serves as our primary, high-speed structural orchestration policy. It models the complete workflow history as a heterogeneous directed graph ($\mathcal{G}_t$) using Graph Attention Networks (GATv2Conv). 
- **The RSL-RL Customization:** Traditional RL libraries (such as Stable-Baselines3) strictly require fixed-dimensional, flat tensor observations, making them incompatible with the dynamic, variable-sized graph structures (`HeteroData`) required by GNNs. 
- To overcome this structural limitation, we customized the high-throughput **RSL-RL** framework. By leveraging PyTorch's `TensorDict` data structures, we packed node features, edge indices, and action masks into a single padded structure. 
- Inside our custom `ActorCriticGNN` policy, we implemented a GPU-optimized `_reconstruct_hetero_batch` algorithm. It dynamically unpads the `TensorDict` on-the-fly, reconstructs the clean heterogeneous graph topology, executes heterogeneous message passing, and pools the final graph embedding to feed the Actor-Critic heads, enabling stable end-to-end PPO training on dynamic graphs.

### System 2: Deliberative Reasoning Fallback (Compliance Guardrail)
Activated as a safety fallback when System 1 confidence $C_t$ falls below the safety threshold ($\tau = 0.85$). 
- **Mechanism:** Inspired by dual-process theories of cognition, it runs an iterative reasoning pipeline utilizing a 3-agent loop (**Generator $\rightarrow$ Verifier $\rightarrow$ Reviser**) to enforce hard constraints and self-correct structural deviations before final execution.
- **Human-in-the-Loop (HITL):** Suspends execution and escalates to a human domain expert if System 2 fails to reach a verified compliant consensus within $N_{max} = 3$ retries.

---

## 2. Repository Structure

```text
rgrl/
├── ai/
│   ├── data/                           # Dataset and clinical scenarios
│   │   ├── clinical_notes/             # Processed clinical notes from MIMIC-IV
│   │   │   ├── data/
│   │   │   └── script/
│   │   ├── clinical_states/            # Definitions of the 35 Semantic Prototypes
│   │   │   └── clinical_states.json
│   │   ├── scenarios/                  # Curated task scenarios
│   │   │   ├── data/
│   │   │   └── script/
│   │   ├── vhas_universe/              # Agent and Tool declarations
│   │   │   └── vhas_universe.json
│   │   ├── workflow_skeletons/         # The 3 Ground-Truth Skeletons
│   │   │   └── workflow_skeletons.json
│   │   └── workflow_traces/            # Raw traces and split JSON files
│   │       ├── cardiovascular/
│   │       ├── gastrointestinal/
│   │       ├── respiratory/
│   │       ├── prepare_splits.py       # Data stratification script
│   │       ├── test_traces.json        # 200 held-out evaluation samples
│   │       ├── train_traces.json       # 1,600 training samples
│   │       └── val_traces.json         # 200 validation samples
│   │
│   ├── dual_system/                    # Inference and Routing implementation
│   │   ├── __init__.py
│   │   ├── deliberative_llm.py         # Generator-Verifier-Reviser pipeline
│   │   ├── neuro_symbolic_router.py    # GNN confidence-based routing wrapper
│   │   └── prompts.py                  # Strict operational guideline definitions
│   │
│   ├── experiments/                    # Official evaluation scripts for thesis metrics
│   │   ├── __init__.py
│   │   ├── evaluate_ablation.py        # System 2 structural ablation study
│   │   ├── evaluate_all_policies.py    # Master script evaluating all 4 policies
│   │   ├── evaluate_topk_ablation.py   # Top-k ablation study for GNN
│   │   └── evaluate_tradeoff.py        # Pareto analysis over tau thresholds
│   │
│   ├── gro/                            # Stage 2: Reinforcement Learning modules
│   │   ├── data/
│   │   │   └── simulation_kb.json      # Deterministic finite-state machine transitions
│   │   ├── gnn/                        # AFAN GNN architecture and PyG env/train
│   │   │   ├── actor_critic_gnn.py     # Custom GNN Actor-Critic policy network
│   │   │   ├── env.py                  # GNN specific episodic environment
│   │   │   ├── train_gnn_deployed.py   # GNN training deployment (Modal)
│   │   │   ├── train.py                # Local training script for GNN
│   │   │   ├── upload_gnn_to_modal.ps1
│   │   │   ├── vec_env_wrapper.py
│   │   │   └── wrappers.py
│   │   ├── scripts/                    # KB building and legacy helpers
│   │   │   ├── build_simulation_kb.py
│   │   │   ├── guidance_dual.py
│   │   │   └── guidance.py
│   │   ├── simulation/                 # Flat vector environment (MLP)
│   │   │   ├── __init__.py
│   │   │   └── env.py
│   │   ├── transformer/                # Sequential policy and sequential env
│   │   │   ├── env.py                  # Transformer specific sequential env
│   │   │   ├── policy_model.py
│   │   │   ├── train_transformer.py    # Local training script for Transformer
│   │   │   ├── train_transformer_deployed.py
│   │   │   └── upload_transformer_to_modal.ps1
│   │   ├── train_mlp.py                # Local training script for MLP
│   │   ├── train_model_a_deployed.py   # Model A training deployment (Modal)
│   │   ├── train_model_b_deployed.py   # Model B training deployment (Modal)
│   │   ├── train_no_guidance_deployed.py # Vanilla RL training deployment (Modal)
│   │   └── train_topk_ablation_deployed.py # Top-k evaluation deployment (Modal)
│   │
│   ├── results/                        # Empirical result logs and visualizations
│   │   ├── figures/                    # Generated high-resolution plots used in the thesis
│   │   │   ├── exp1_pareto_tradeoff.png
│   │   │   ├── exp2_ablation_study.png
│   │   │   ├── guideline_compliance_boxplot.png
│   │   │   ├── learning_curves_2.png
│   │   │   └── tsne_visualization.png
│   │   ├── ablation_topk.csv           # Table 3.3 raw data
│   │   ├── main_evaluation_summary.csv # Table 3.4 raw data
│   │   ├── pareto_tradeoff_results.csv # Figure 1 raw data
│   │   ├── plot_training_results.py    # Script to programmatically regenerate figures
│   │   ├── system2_ablation_results.csv # Figure 2 raw data
│   │   ├── test_eval_200_episodes.csv  # Boxplot raw data
│   │   ├── training_dynamics.csv       # Learning curves raw data
│   │   ├── tsne_embeddings_2d.csv      # t-SNE coordinates
│   │   └── wrl_nearest_neighbors.json  # Table 3.2 raw data
│   │
│   ├── models/                         # Trained model artifacts and weights
│   │   ├── stage1_wrl_encoders/        # Fine-tuned State and Action encoders
│   │   └── stage2_gro_policies/        # SB3 (.zip) and RSL-RL/PyG (.pt) checkpoints
│   │
│   └── rsl_rl/                         # Customized RSL-RL framework supporting TensorDict
│
├── web/                                # Next.js Progressive Disclosure Frontend
│   ├── app/
│   ├── backend/                        # FastAPI TraceStore service
│   │   ├── api.py                      # REST endpoints for trace queries
│   │   ├── models.py                   # SQLAlchemy database schemas
│   │   ├── requirements.txt
│   │   ├── seed_tracestore.py          # Script to populate the PostgreSQL database
│   │   └── store.py                    # VHASTraceStore database driver
│   ├── components/                     # UI components (vhas)
│   ├── data/                           # OTLP demo traces
│   ├── lib/
│   └── public/
│
└── docker-compose-vhas-db.yml          # Docker Compose configuration for PostgreSQL
```

---

## 3. Experimental Setup

The evaluation is strictly bounded to emergency internal medicine triage across three typical clinical pathways: **Monitor & Release**, **Quick Intervention**, and **Complex Care Loop**.
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
- OpenAI API Key (configured in `.env.local` for System 2 evaluation)

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

**1. Run Pareto Trade-off Analysis ($\tau$ Sweep):**
```bash
python ai/experiments/evaluate_tradeoff.py
```
*This script iterates through $\tau$ values, prints GNN/LLM logs to the terminal, and exports metrics to `ai/results/pareto_tradeoff_results.csv`.*

**2. Run System 2 Ablation Study:**
```bash
python ai/experiments/evaluate_ablation.py
```
*This script compares policy configurations and exports metrics to `ai/results/system2_ablation_results.csv`.*

**3. Run Complete Policy Evaluation (Table 3.4 & Boxplot generation):**
```bash
python ai/experiments/evaluate_all_policies.py
```
*This script evaluates all 4 trained policy checkpoints over 200 episodes, exporting boxplot data to `ai/results/test_eval_200_episodes.csv` and summary table data to `ai/results/main_evaluation_summary.csv`.*

**4. Run Top-K Ablation Study (Table 3.3):**
```bash
python ai/experiments/evaluate_topk_ablation.py
```
*This script evaluates GNN behavior under different $k$ masking thresholds, exporting data to `ai/results/ablation_topk.csv`.*

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