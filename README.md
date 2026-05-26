# Representation-Guided Reinforcement Learning for Guideline-Compliant Multi-Agent Orchestration

This repository contains the official implementation of the **VHAS (Vietnam Health-Agent System)**, a testbed designed to evaluate the **Representation-Guided Reinforcement Learning (RGRL)** framework. 

The primary objective of this work is to address the combinatorial explosion of state and action spaces in multi-agent reinforcement learning (RL) when operating in environments characterized by high-dimensional, unstructured text inputs (Semantic Variability). The proposed system enforces strict adherence to predefined operational protocols (Guideline Compliance) via a Neuro-Symbolic Dual-System architecture.

---

## 1. System Architecture

The RGRL framework is implemented in two offline-online stages, integrating a dual-system inference routing mechanism during deployment:

<p align="center">
  <img src="ai/results/figures/overall_framework.jpg" width="85%" alt="RGRL Overall Framework" />
</p>

### Stage 1: Workflow Representation Learning (WRL)
- **Methodology:** A Dual-Encoder architecture optimized via self-supervised Contrastive Learning (Multiple Negatives Ranking Loss).
- **Function:** Maps highly variable textual contexts (Semantic States) and static entity specifications (Agents/Tools) into a unified, continuous embedding space, forming a static Knowledge Map.

### Stage 2: Guided Reinforcement Optimization (GRO)
- **Methodology:** Proximal Policy Optimization (PPO) combined with dynamic Action Masking.
- **Function:** At each timestep, the system queries the Knowledge Map to retrieve top-$k$ relevant entities, generating a binary mask. This masks invalid logits before the Softmax layer, pruning the action space and stabilizing RL convergence.

---

### System 1: The Agent-Flow Attention Network (AFAN) & RSL-RL Integration
System 1 serves as our primary, high-speed structural orchestration policy. It models the complete workflow history as a heterogeneous directed graph ($\mathcal{G}_t$) using Graph Attention Networks (GATv2Conv). 

<p align="center">
  <img src="ai/results/figures/afan_architecture.jpg" width="90%" alt="AFAN GNN Architecture" />
</p>

- **The RSL-RL Customization:** Traditional RL libraries (such as Stable-Baselines3) strictly require fixed-dimensional, flat tensor observations, making them incompatible with the dynamic, variable-sized graph structures (`HeteroData`) required by GNNs. 
- To overcome this structural limitation, we customized the high-throughput **RSL-RL** framework. By leveraging PyTorch's `TensorDict` data structures, we packed node features, edge indices, and action masks into a single padded structure. 
- Inside our custom `ActorCriticGNN` policy, we implemented a GPU-optimized `_reconstruct_hetero_batch` algorithm. It dynamically unpads the `TensorDict` on-the-fly, reconstructs the clean heterogeneous graph topology, executes heterogeneous message passing, and pools the final graph embedding to feed the Actor-Critic heads, enabling stable end-to-end PPO training on dynamic graphs.

---

### System 2: Deliberative Reasoning Fallback (Compliance Guardrail)
Activated as a safety fallback when System 1 confidence $C_t$ falls below the safety threshold ($\tau = 0.85$). 

<p align="center">
  <img src="ai/results/figures/dual_system_router.jpg" width="85%" alt="Dual-System Router" />
</p>

- **Mechanism:** Inspired by dual-process theories of cognition, it runs an iterative reasoning pipeline utilizing a 3-agent loop (**Generator $\rightarrow$ Verifier $\rightarrow$ Reviser**) to enforce hard constraints and self-correct structural deviations before final execution.

<p align="center">
  <img src="ai/results/figures/deliberative_pipeline.jpg" width="80%" alt="Deliberative Reasoning Pipeline" />
</p>

- **Human-in-the-Loop (HITL):** Suspends execution and escalates to a human domain expert if System 2 fails to reach a verified compliant consensus within $N_{max} = 3$ retries.

---

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

### Running Evaluation Experiments
To execute the actual evaluation logic over the held-out Test Set (200 cases) using the real inference pipelines:

**1. Execute Pareto Trade-off Analysis ($\tau$ Sweep):**
```bash
python ai/experiments/evaluate_tradeoff.py
```
*This script iterates through $\tau$ values, prints GNN/LLM logs to the terminal, and exports metrics to `ai/results/pareto_tradeoff_results.csv`.*

<p align="center">
  <img src="ai/results/figures/exp1_pareto_tradeoff.png" width="70%" alt="Pareto Tradeoff" />
</p>

**2. Execute System 2 Ablation Study:**
```bash
python ai/experiments/evaluate_ablation.py
```
*This script compares policy configurations and exports metrics to `ai/results/system2_ablation_results.csv`.*

<p align="center">
  <img src="ai/results/figures/exp2_ablation_study.png" width="75%" alt="System 2 Ablation Study" />
</p>

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

<p align="center">
  <img src="ai/results/figures/learning_curves_2.png" width="80%" alt="Learning Curves" />
  <img src="ai/results/figures/guideline_compliance_boxplot.png" width="45%" alt="Guideline Compliance Boxplot" />
</p>

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

<p align="center">
  <img src="ai/results/figures/vhas_web_ui_demo.jpg" width="90%" alt="VHAS Web UI Demo" />
</p>

---