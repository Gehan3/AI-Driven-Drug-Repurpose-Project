# AI-Driven-Drug-Repurpose-Project
This project focuses on AI-driven drug repurposing using X-Gboost &amp; Graph Neural Networks (GNNs). The system is designed to model complex biological entities as graphs and predict new drug-disease associations by analyzing network-based relationships, can be used in app like precision medicine and identifying  new therapeutic indications for existing, approved drugs — cutting the cost, time, and clinical risk of conventional drug development. This project implements a dual-model AI framework that combines **DREAMwalk** (semantic random-walk graph embeddings + XGBoost) and **TxGNN** (a heterogeneous graph neural network), both trained independently on the **Hetionet** biomedical knowledge graph, and unified through a weighted ensemble. Every prediction is made explainable through SHAP-based feature attribution and graph-native explanation methods.

## 🌐 Live Demo

- **Try the App:** [https://perfectteam.doctorgehan3.workers.dev/](https://perfectteam.doctorgehan3.workers.dev/)

## 🎬 Demo Video
Watch the full demo on YouTube: [AI-Driven Drug Repurposing Demo](https://www.youtube.com/watch?v=HsqOtiYIFGs)
![Alt Text](https://github.com/Gehan3/AI-Driven-Drug-Repurpose-Project/blob/main/cover.png)

Drug repurposing 
> 📄 Full thesis: [AI_Driven_Drug_Repurpose.pdf](./AI_Driven_Drug_Repurpose.pdf) · 🖼️ Poster: [AI_Drug_Repurposing_Poster.pdf](./AI_Drug_Repurposing_Poster%20.pdf)
---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Architecture](#architecture)
- [Dataset — Hetionet v1.0](#dataset--hetionet-v10)
- [Models](#models)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Results](#results)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Current Progress](#current-progress)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Systematic drug repurposing is hard because:
- Known drug–disease associations are **extremely sparse** (only 0.36% of the compound–disease search space in Hetionet is a known treatment).
- Biomedical data is **highly heterogeneous** (29 source databases, 11 entity types, 24 relation types).
- Most predictive models are **black boxes**, offering no biological rationale for a prediction.

This framework addresses all three by combining two architecturally independent models, enforcing a **leakage-free evaluation protocol**, and integrating a full **explainability layer** (SHAP + graph-native explanations) so every ranked candidate comes with a checkable reason.

## Why This Matters

- **Lower cost & risk** — repurposed drugs already have known safety/toxicity profiles, shortening the regulatory path.
- **Rapid response** — computational repurposing enables fast candidate generation during health emergencies (e.g., COVID-19).
- **Model diversity reduces false positives** — requiring agreement between a walk-based model (DREAMwalk) and a message-passing model (TxGNN) filters out spurious single-model predictions.
- **Interpretability** — SHAP and metapath/relation-ablation explanations turn "black-box" scores into traceable biological hypotheses.

## Architecture

```
Hetionet v1.0 (47,031 nodes · 2,250,197 edges)
        │
        ▼
 Leakage-Free Split — strip test-fold Compound-treats-Disease (CtD)
 edges BEFORE any embedding, random walk, or graph construction
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                              ▼                              
 BRANCH A: DREAMwalk               BRANCH B: TxGNN                  
 Sparse CSR graph                  PyG HeteroData construction      
   → Vectorized Jaccard similarity   → 2× HeteroConv (SAGEConv)     
   → Semantic teleport random walk   → Two-phase training           
     (network traversal ⇄            (pretrain on all relations,   
      semantic teleportation)         fine-tune on CtD)             
   → Heterogeneous Skip-gram         → DistMult link scoring        
     node embeddings (128-d)                                        
   → XGBoost classifier (256-d                                      
     concatenated pair vectors)                                     
        │                              │
        └──────────────┬───────────────┘
                        ▼
         Weighted Ensemble Scoring
         55% DREAMwalk/XGBoost + 45% TxGNN
                        │
                        ▼
         Ranked, Explainable Candidate List
         (SHAP + metapath evidence + graph-native
          explanations for TxGNN)
```

## Dataset — Hetionet v1.0

| Property | Value |
|---|---|
| Source databases integrated | 29 |
| Node types (metanodes) | 11 |
| Relation types (metaedges) | 24 |
| Total nodes | 47,031 |
| Total relationships | 2,250,197 |
| Compounds (drugs) | 1,552 |
| Diseases | 137 |
| Genes | 20,945 |
| Pathways | 1,822 |
| Known Compound-treats-Disease pairs | 755 (≈0.36% of the full search space) |

## Models

### DREAMwalk + XGBoost
- Semantic information-guided, teleport-based random walks combining local graph traversal with similarity-based teleportation across drug/disease nodes.
- Transition and teleport probabilities derived from vectorized Jaccard similarity and ontology-based (ATC / MeSH / DO) information-content similarity.
- Heterogeneous Skip-gram (type-aware negative sampling) produces 128-dimensional embeddings.
- Drug–disease pairs represented as 256-dimensional concatenated embeddings, classified with XGBoost (300 estimators, max depth 6, `hist` tree method).
- Evaluated with a **leak-free 10-fold stratified cross-validation** protocol — all test-fold CtD edges are removed before similarity matrices or walks are (re)computed each fold.

### TxGNN
- Heterogeneous graph neural network trained end-to-end in PyTorch Geometric.
- 128-dimensional type-specific learnable node embeddings → two-layer `HeteroConv` encoder with relation-specific `SAGEConv` operators → DistMult link scoring for CtD/CpD relations.
- Two-phase training: pretraining across **all** relation types, then fine-tuning on CtD alone, with `BCEWithLogitsLoss` and leakage-aware dynamic negative sampling.
- Evaluated with **leak-free 5-fold stratified cross-validation**; benchmarked against RGCN, DistMult, and MLP baselines trained under an identical harness.

### Ensemble
Final confidence score = **0.55 × DREAMwalk/XGBoost + 0.45 × TxGNN**, exploiting the architectural diversity between walk-based co-occurrence learning and iterative graph message passing to reduce false positives.

## Explainable AI (XAI)

| Layer | Applies to | Method |
|---|---|---|
| Global & local feature attribution | XGBoost | TreeSHAP (waterfall, force, summary, beeswarm, dependence plots) |
| Biological corroboration | XGBoost | Metapath evidence — shared genes / pathways along the Compound↔Disease path in Hetionet |
| Embedding-dimension attribution | TxGNN | Exact decomposition of the DistMult score into its 128 dimension-level terms |
| Neighbor saliency | TxGNN | Gradient-based backpropagation to real, trained-on 1-hop graph neighbors |
| Relation-type importance | TxGNN | Leave-one-relation-out ablation, restored after each measurement |
| Aggregated importance | TxGNN | Cross-prediction heatmaps of dimension and relation-type contribution |

All explanations use real Hetionet/DrugBank entity names — never raw node IDs — and are reported honestly, including where the two independent runs / methods **disagree**, rather than only where they agree.

## Results

| Metric | DREAMwalk + XGBoost (10-fold CV) | TxGNN (5-fold CV) |
|---|---|---|
| Mean AUROC | 0.9396 | 0.9826 |
| Mean AUPR | 0.9378 | 0.9869 |
| Mean Accuracy | 0.8775 | 0.9342 |

The DREAMwalk/XGBoost leakage-free reimplementation (AUROC 0.9396 / AUPR 0.9378 / Accuracy 0.8775) closely matches the original DREAMwalk paper's reported average across three knowledge graphs (AUROC 0.938 / AUPR 0.939 / Accuracy 0.873), confirming the reimplementation is faithful despite the stricter, leakage-free protocol.

Real-world case studies on **Alzheimer's Disease** and **Breast Carcinoma** demonstrate practical applicability, and the top-ranked novel candidates are cross-checked against real Hetionet gene/pathway evidence rather than reported as bare scores.

## Deployment

A three-tier web application decouples UI, orchestration, and inference:

| Layer | Stack | Role |
|---|---|---|
| Frontend | React (SPA) | Search a drug/disease, choose a model, view ranked predictions, evidence, and an interactive relationship graph |
| Backend gateway | Node.js / Express | Single entry point for the frontend; validates requests, forwards to the prediction service, reshapes responses |
| Prediction service | Python / FastAPI | Wraps the trained XGBoost/DREAMwalk, TxGNN, and ensemble models plus the literature-evidence pipeline |

In development, Express runs on port `8000` and FastAPI on port `8001`, started together via a single launcher script.

## Repository Structure

> Adjust this section to match the actual repo layout.

```
.
├── data/                     # Hetionet parquet/TSV files, processed splits
├── notebooks/                # Exploratory analysis & model development notebooks
├── src/
│   ├── dreamwalk/             # Random walk, Jaccard similarity, Skip-gram, XGBoost pipeline
│   ├── txgnn/                 # HeteroData construction, TxGNN model, baselines, explainer
│   ├── ensemble/               # Weighted ensemble scoring
│   └── xai/                    # SHAP suite + metapath evidence + TxGNN explainability
├── backend/
│   ├── gateway/                # Node.js / Express API gateway
│   └── prediction-service/     # FastAPI inference service
├── frontend/                  # React application
├── docs/
│   ├── AI_Driven_Drug_Repurpose.pdf   # Full thesis
│   └── AI_Drug_Repurposing_Poster.pdf # Conference poster
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<org>/<repo>.git
cd <repo>

# Backend — prediction service (FastAPI)
cd backend/prediction-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# Backend — gateway (Express)
cd ../gateway
npm install
npm start   # runs on port 8000

# Frontend (React)
cd ../../frontend
npm install
npm start
```

> Model weights and the processed Hetionet graph are not committed to the repository due to size — see `docs/` for download/regeneration instructions.

## Current Progress

**RAG-Based Literature Retrieval** — a Retrieval-Augmented Generation pipeline is under active development to retrieve relevant PubMed literature as biological evidence for predicted drug–disease associations, grounding each prediction in published research.

**Dynamic Knowledge Graph Enrichment** — a Neo4j-backed pipeline that continuously extends the static Hetionet graph with new drug–gene and gene–disease relationships extracted from recent PubMed publications, using entity/relationship deduplication before any incremental update.

**Conference Submission** — the accompanying paper, *"Vectorized, Leak-Free Biomedical Knowledge Graph Learning for Drug Repurposing: A Computationally Scalable Extension of DREAMwalk,"* has been submitted to **NILES2026** (Paper #1571325000) and is currently **under review**.

## Future Work

- Extend evaluation to additional knowledge graphs (MSI, KEGG) for cross-KG validation.
- Incorporate multi-omics data (genomic, transcriptomic, proteomic) for richer node features.
- Explore more advanced heterogeneous GNN architectures and attention-based ensemble weighting.
- Complete the RAG-based PubMed evidence retrieval and dynamic knowledge-graph enrichment pipeline.
- Pursue in-vitro / in-vivo validation of the top-ranked repurposing candidates.
- Package the platform as a **SaaS product**, opening it up to subscription-based access for researchers, clinicians, and institutions.

Submitted in partial fulfillment of the **Digilians 9-Month Diploma in Applied AI and Data Analytics**, August 2026, awarded by the Military Technical College.

## References

1. Bang, D., Lim, S., Lee, S., & Kim, S. (2023). *Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers* (DREAMwalk). **Nature Communications**.
2. Huang, K., & Zitnik, M. (2024). *A foundation model for clinician-centered drug repurposing* (TxGNN).
3. Zeng, X. et al. (2019). *deepDR: a network-based deep learning approach to in silico drug repositioning*.
4. Wang, Y. et al. (2022). *DrugRepo: a novel approach to repurposing drugs based on chemical and genomic features*.
5. Amiri, Razmara, Parvizpour & Izadkhah (2023). *IDDI-DNN: a novel efficient drug repurposing framework via drug–disease association data integration using CNNs*.
6. Talevi, A., & Bellera, C. (2020). *Drug repurposing strategies, challenges, and opportunities*.

## License

Add a license (e.g., MIT, Apache-2.0) before publishing this repository publicly, and update this section accordingly.

---

*This is an academic research project. Predictions are computational hypotheses for experimental follow-up, not clinical recommendations.*

