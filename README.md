# NeuroSymbolic AI — Healthcare Clinical Decision Assistant

 

---

[View colab notebook](https://colab.research.google.com/drive/1qn9tLHnWus-UnTxWiR4rnxjWQdNMREZR?usp=sharing)

##  Overview

Pure LLMs fail in high-stakes medical settings — they hallucinate diagnoses, ignore clinical rules, and waste context on irrelevant information. This project implements **Adaptive Symbolic Grounding (ASG)**, a neurosymbolic architecture where the Knowledge Graph and LLM collaborate: the symbolic layer activates exactly when the neural model needs grounding.

> *"A system that knows when it doesn't know — and uses symbolic AI to fill the gap."*

### Problem Statement 2 — syNNapse'26
**Use Case:** Healthcare Clinical Decision Assistant  
**Methodology:** Ontologies + First-Order Logic + Context Management  
**Evaluation:** Model Benchmark → Agent Benchmark → ContextBench

---

##  Key Differentiators

| Feature | Standard RAG | Our System |
|---|---|---|
| Ontology retrieval | Static — always retrieves | **Adaptive** — only when LLM uncertain |
| Rule checking |  None | First-Order Logic rules from ontology |
| Context management | Fixed window | **3 policies** with budget scheduler |
| Hallucination handling | Post-hoc filtering | **Symbolic grounding prevents it** |
| Explainability | Black box | **Confidence score + rule status + matched symptoms** |
| Benchmarks | Custom only | **TruthfulQA + MMLU + BIG-Bench** (published) |

---

##  Architecture

### System Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│     Symbolic Pre-Filter         │  ← Symptom extraction, intent classification
│     (spaCy / substring match)   │    constraint detection from query
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌─────────────────────┐
│ KG Lookup +  │  │ Adaptive Context    │
│ Rule Checker │  │ Policy Engine       │
│ (NetworkX)   │  │ (Full/RAG/Compress) │
└──────┬───────┘  └────────┬────────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
    ┌────────────────────────┐
    │  Symbolic Scaffold     │  ← Natural language rendering of KG results
    │  Injection             │    "Candidate: Scabies (confidence: 2.16),
    └────────────┬───────────┘     Rule: itching AND night → SATISFIED"
                 ▼
    ┌────────────────────────┐
    │  LLM Reasoning Core    │  ← Mistral-7B (QLoRA fine-tuned)
    │  (Mistral-7B / Llama)  │    Reasons OVER verified symbolic scaffold
    └────────────┬───────────┘
                 ▼
    ┌────────────────────────┐
    │  Symbolic Verifier     │  ← Constraint check on output
    └────────────┬───────────┘
                 ▼
         Verified Output
   (Disease + Confidence + Rule Status + Treatment)
```

### Three-Layer Design

```
Layer 1 — Neural Core
├── Model       : Mistral-7B-Instruct-v0.3
├── Training    : QLoRA (r=16, alpha=16) via Unsloth
├── Dataset     : Custom medical symptom-disease-treatment JSON
└── Why Mistral : Strong instruction-following, Unsloth official support,
                  fits T4 GPU in 4-bit (~5.5 GB VRAM)

Layer 2 — Symbolic Engine
├── Knowledge Graph : NetworkX DiGraph from ontology.json
├── Node types      : disease, symptom, treatment, risk, rule
├── Edge types      : has_symptom (weighted), symptom_of, has_treatment, has_rule
├── Reasoning       : symptom→disease lookup + First-Order Logic rule checking
└── Why NetworkX    : Pure Python, no external DB, fast in-memory traversal

Layer 3 — Context Optimizer
├── Policy 1 : full_context (2048 tokens) — entire history + ontology
├── Policy 2 : rag_policy (1024 tokens)  — targeted KG retrieval
├── Policy 3 : compression (512 tokens)  — compressed + minimal ontology
└── Why 3 policies : Directly maps to ContextBench evaluation requirement
```

---

##  Benchmarks

This project runs **three evaluation phases** as required by the problem statement.

### Phase 1 — Model Benchmark

Compares base Mistral-7B against our QLoRA fine-tuned (domain-pretrained) model.


**Benchmarks used:**

| Benchmark | What it tests | Why we used it |
|---|---|---|
| **Domain QA** (custom) | Symptom → disease + treatment accuracy | Directly measures medical domain adaptation |
| **TruthfulQA** | Hallucination detection, uncertainty awareness | PS explicitly required; tests if model says "I don't know" vs confidently wrong |
| **MMLU Clinical Knowledge** | Medical domain factual knowledge (MCQ) | Published academic benchmark — judges can independently verify |
| **BIG-Bench Boolean** | Logical reasoning ability | PS required; tests whether model can apply AND/OR logic — critical for rule-based diagnosis |

**Metrics:**

| Metric | Meaning |
|---|---|
| Diagnosis Accuracy | % of cases where correct disease was identified |
| Hallucination Rate | % of cases where model stated wrong disease confidently |
| Treatment Keyword Score | % of expected treatment terms present in response |
| TruthfulQA Accuracy | % of truthful/uncertainty-aware answers |
| MMLU Clinical Accuracy | % correct on published clinical MCQ |
| BIG-Bench Accuracy | % correct on boolean logic tasks |

#### Model Benchmark Results

<img width="1073" height="110" alt="Screenshot 2026-04-14 233844" src="https://github.com/user-attachments/assets/f34e2729-e925-4dba-bc8b-02e7fa0dc99e" />

<img width="828" height="334" alt="image" src="https://github.com/user-attachments/assets/ba9e679c-4864-43cd-be15-82157b0c4a37" />




 

 

---

### Phase 2 — Agent Benchmark

Compares a **Baseline LLM-only agent** against our **Neurosymbolic agent** (LLM + Knowledge Graph + Rules).

**Why LangChain:** PS explicitly mentions LangChain. It provides `AgentExecutor` for tool-calling loops, built-in conversation memory, and the cleanest baseline↔neurosymbolic swap — both agents use identical model weights, only the tools differ. This ensures improvements come from symbolic grounding, not model differences.

**Task Types (6 types, 13 tasks):**

| Type | Description | Example |
|---|---|---|
| Single-hop diagnosis | 2-3 clear symptoms → 1 disease | Itching at night → Scabies |
| Multi-hop treatment | Symptom → Disease → Treatment chain | Headache + confusion → Subdural → CT + surgery |
| Differential diagnosis | Ambiguous symptoms, pick best | Abdominal pain + bleeding → multiple candidates |
| Rule-based reasoning | Apply formal clinical rules | Headache AND confusion → rule SATISFIED |
| Complex multi-symptom | 5+ symptoms, prioritize | Emergency case with 6 symptoms |
| Treatment planning | Given diagnosis → full plan | Scabies confirmed → household precautions |

**Metrics:**

| Metric | Meaning |
|---|---|
| Disease Accuracy | Correct diagnosis rate |
| Treatment Score | Expected treatment keywords found |
| Hallucination Rate | Confident wrong diagnosis rate |
| Rule Adherence | Correctly applied ontology logic rules |
| Reasoning Depth | Depth of clinical reasoning (0–3 scale) |
| Grounding Score | How much output uses ontology-verified facts (0–1) |

#### Agent Benchmark Results

 <img width="1183" height="119" alt="Screenshot 2026-04-14 233955" src="https://github.com/user-attachments/assets/acb4d45e-9de1-496d-98bd-8dc5f8154fb1" />


<img width="1646" height="531" alt="image" src="https://github.com/user-attachments/assets/458719d3-da03-4b7c-a003-542adf2c7acb" />
<img width="1645" height="449" alt="image" src="https://github.com/user-attachments/assets/192550de-9cb1-4054-a147-785e529aed3b" />



 

 

---

### Phase 3 — ContextBench

Evaluates how efficiently each system manages context under token and cost constraints.

> **Note:** The official ContextBench GitHub repo (`ContextBench/contextbench`) contains only a website with no runnable code. We implemented the complete framework from scratch, strictly following the PS specification — config YAMLs, 3 policies, 6 metrics, 4-system comparison. This is intentionally more transparent than using a black-box tool.

**Experiment Configuration (YAML-based, as PS requires):**

```yaml
# configs/rag_policy.yaml
experiment_name: contextbench_rag_policy
use_case: Healthcare Clinical Decision Assistant
model: mistral-7b-instruct-v0.3
dataset: patient_records_and_clinical_guidelines
policy: rag_policy
policy_description: Only retrieve relevant ontology subgraph for current query
token_budget: 1024
cost_per_1k_tokens: 0.0002
systems:
  - base_model
  - pretrained_model
  - baseline_agent
  - neurosymbolic_agent
```

**3 Context Policies:**

| Policy | Token Budget | Strategy | Why |
|---|---|---|---|
| `full_context` | 2048 | Entire patient history + full ontology | Highest accuracy baseline |
| `rag_policy` | 1024 | Targeted KG retrieval per query | Neurosymbolic advantage: precise retrieval |
| `compression_policy` | 512 | Compressed history + minimal ontology | Stress test — who degrades least? |

**6 Metrics (PS Required):**

| # | Metric | Formula | Meaning |
|---|---|---|---|
| 1 | **Hallucination** | `1 if wrong AND confident else 0` | Confident wrong diagnosis = dangerous |
| 2 | **Task Success Rate** | `correct / total × 100` | % correct diagnoses |
| 3 | **Token Usage** | `input_tokens + output_tokens` | Compute cost proxy |
| 4 | **Cost per Query** | `(tokens / 1000) × $0.0002` | Deployment cost estimate |
| 5 | **Latency** | `time.time()` delta (seconds) | Response speed |
| 6 | **Context Efficiency** *(novel)* | `task_success / (tokens_used / budget)` | Accuracy per token budget fraction |

> **Context Efficiency** is our novel contribution — PS named it but provided no formula. Higher score = better accuracy with less budget used. Example: 80% accuracy using 50% of budget → CE = 1.6 vs 80% accuracy using 90% → CE = 0.89.

#### ContextBench Results

<img width="1072" height="376" alt="Screenshot 2026-04-14 234205" src="https://github.com/user-attachments/assets/1fdd6e5c-1a5e-4c52-b475-2d3c14fd7d9c" />


 <img width="1628" height="437" alt="Screenshot 2026-04-14 234306" src="https://github.com/user-attachments/assets/583fda03-babe-4cab-ae57-03c905d2ff33" />

**PS-Required Table (System | Accuracy | Tokens | Cost):**

 <img width="1061" height="171" alt="Screenshot 2026-04-14 234143" src="https://github.com/user-attachments/assets/d3cbaed9-b635-42d0-bdb4-67a1a416360e" />


 

 
 
```

---

##  Setup & Running

### Prerequisites

```bash
# Python 3.10+
# Google Colab (recommended) — T4 GPU runtime
# Google Drive — for checkpoint saving
```

### Step 1 — Model Benchmark

Open `syNNapse26_ModelBenchmark_FINAL.ipynb` in Google Colab:

```
Runtime → Change runtime type → T4 GPU
Run All (Ctrl+F9)
```

This will:
1. Install all dependencies (Unsloth, HuggingFace, TRL)
2. Load Mistral-7B-Instruct in 4-bit quantization
3. Evaluate base model on medical domain QA
4. Fine-tune using QLoRA on your `training_data.json`
5. Evaluate pretrained model on same data
6. Run TruthfulQA, MMLU Clinical, BIG-Bench benchmarks
7. Generate comparison plots and save to Drive

### Step 2 — Agent Benchmark

Open `syNNapse26_AgentBenchmark_FINAL.ipynb`:

```
Ensure training_data.json and ontology.json are in Drive
Run All
```

### Step 3 — ContextBench

Open `syNNapse26_ContextBench_FINAL.ipynb`:

```
Ensure previous notebooks have been run (results on Drive)
Run All
```

### GPU Crash Recovery

All notebooks save results to Google Drive after each major step. If GPU resets:

```python
# Model Benchmark: restart from Cell 11 (reload LoRA from Drive)
# Agent Benchmark: restart from Cell A3 (reload model)  
# ContextBench: restart from Cell CB5 (resume from last saved policy)
```

---

##  Tech Stack

### Core ML

| Tool | Version | Why we used it |
|---|---|---|
| **Unsloth** | Latest | 60% less VRAM than standard HuggingFace. Mistral-7B fits T4 GPU (15GB) in 4-bit (~5.5GB). Official Colab support. Without it, full fine-tuning is impossible on free Colab. |
| **Mistral-7B-Instruct-v0.3** | v0.3 | Strong instruction-following, beats Llama-2-13B on many tasks at half the size. PS mentions Mistral/Llama. Unsloth has native optimization for it. |
| **QLoRA** (via PEFT) | 4-bit | Quantized LoRA — trains only ~0.5% of parameters (adapters), rest frozen. Makes continued pretraining feasible without A100. `r=16` rank chosen as sweet spot for T4. |
| **TRL / SFTTrainer** | Latest | Supervised Fine-Tuning Trainer — purpose-built for instruction datasets. Less boilerplate than raw HuggingFace `Trainer`. 8-bit AdamW optimizer halves memory for optimizer states. |
| **BitsAndBytes** | Latest | Actual 4-bit quantization kernel. Unsloth uses this internally to compress model weights 4x. |

### Symbolic / Knowledge Graph

| Tool | Version | Why we used it |
|---|---|---|
| **NetworkX** | 3.x | Pure Python in-memory directed graph. No external database needed. Fast subgraph traversal for symptom→disease lookup. Supports weighted edges (symptom probabilities). |
| **Ontology (custom JSON)** | — | Disease-symptom-treatment-rule structure. Rules stored as First-Order Logic strings (`"symptom_A AND symptom_B"`). Enables formal constraint checking without an SMT solver. |

### Agent Framework

| Tool | Version | Why we used it |
|---|---|---|
| **LangChain** | Latest | PS explicitly mentions LangChain. Provides `AgentExecutor` for tool-calling loops. Custom LLM wrapper allows our Unsloth model inside LangChain ecosystem. Both baseline and neurosymbolic agents built with same framework — fair comparison. |

### Benchmarking

| Benchmark | Source | Why we used it |
|---|---|---|
| **TruthfulQA** | HuggingFace `truthful_qa` | PS explicitly required. Published research benchmark. Tests hallucination and epistemic humility — critical for medical AI. |
| **MMLU Clinical Knowledge** | HuggingFace `cais/mmlu` | Published benchmark. Clinical knowledge MCQ subset directly relevant to healthcare use case. 265 test questions. |
| **BIG-Bench Boolean** | HuggingFace `lighteval/big_bench_hard` | PS required. Tests logical reasoning — essential for rule-based diagnosis (AND/OR logic). |
| **ContextBench (custom)** | Implemented from scratch | PS specified framework but GitHub repo has no runnable code. We implemented exact spec: YAML configs, 3 policies, 6 metrics, 4-system table. |

### Data & Visualization

| Tool | Why we used it |
|---|---|
| **Seaborn** | Publication-quality statistical plots. `whitegrid` theme, grouped bar charts, heatmaps, radar charts in fewer lines than raw matplotlib. |
| **Pandas** | DataFrame manipulation for aggregating multi-system results. Styled tables with `highlight_max/min` for direct visual comparison. |
| **PyYAML** | Experiment configuration files in YAML format — exactly as PS specifies "Create an experiment configuration file". |
| **PyTorch** | Foundation for all GPU operations. `torch.no_grad()` for memory-efficient inference. `cuda.empty_cache()` for GPU crash prevention. |

---

## Ontology Format

Your `ontology.json` should follow this structure:

```json
{
  "Scabies": {
    "symptoms": {
      "intense itching": 0.74,
      "especially at night": 0.92,
      "small blisters or bumps": 0.84
    },
    "risk_factors": {
      "general risk": 0.5
    },
    "treatment": [
      "Prescription medications (topical or oral scabicides)",
      "washing clothes and bedding in hot water",
      "vacuuming and cleaning home"
    ],
    "rules": [
      "intense itching AND especially at night",
      "especially at night AND small blisters or bumps"
    ]
  }
}
```

**Fields:**
- `symptoms` — symptom name → probability weight (0-1). Higher = stronger indicator.
- `risk_factors` — known risk factors with probability.
- `treatment` — list of evidence-based treatments.
- `rules` — First-Order Logic rules (`AND` operator). If all parts match → rule satisfied → +0.5 confidence bonus.

---

##  Training Data Format

Your `training_data.json` should be a list of conversation objects:

```json
[
  {
    "messages": [
      {"role": "system",    "content": "You are an AI medical assistant."},
      {"role": "user",      "content": "Symptoms: rash, itching, swelling | History: general risk"},
      {"role": "assistant", "content": "Disease: Food Allergy\nReasoning: rash, itching are strong indicators...\nTreatment: Avoidance of allergenic food, antihistamines, epinephrine"}
    ]
  }
]
```

---

##  ContextBench — Novel Metric Explanation

### Context Efficiency Formula

```
Context Efficiency = Task Success / (Token Usage / Token Budget)
```

**Why this metric matters:**

A system that achieves 80% accuracy while using 50% of its token budget is **more valuable** than one achieving 80% accuracy using 90% of the budget — especially for real-world deployment where inference costs are cumulative.

```
Example:
  System A: accuracy=0.80, tokens=1800, budget=2048
  → CE = 0.80 / (1800/2048) = 0.80 / 0.88 = 0.91

  System B (Neurosymbolic): accuracy=0.80, tokens=1024, budget=2048  
  → CE = 0.80 / (1024/2048) = 0.80 / 0.50 = 1.60  ← 75% more efficient
```

The neurosymbolic agent scores higher on Context Efficiency because the Knowledge Graph does the compression intelligently — only relevant ontology facts reach the LLM, reducing token waste without sacrificing accuracy.

---

##  Key Results Summary

<img width="1072" height="376" alt="Screenshot 2026-04-14 234205" src="https://github.com/user-attachments/assets/150600b6-d3da-4354-a9d3-6c5041202e9e" />



---


##  License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — QLoRA fine-tuning optimization
- [HuggingFace](https://huggingface.co) — Model hub, Transformers, Datasets
- [LangChain](https://github.com/langchain-ai/langchain) — Agent framework
- [NetworkX](https://networkx.org) — Knowledge graph implementation
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) — Hallucination benchmark
- [MMLU](https://github.com/hendrycks/test) — Academic benchmark
- [BIG-Bench](https://github.com/google/BIG-bench) — Reasoning benchmark

---

<div align="center">

**Built for syNNapse'26 · Problem Statement 2 · Healthcare Neurosymbolic AI**

*Grounded · Explainable · Hallucination-Free*

</div>
