# BDH-Inspired Plastic Memory Model for Long-Form Narrative Consistency

## Overview

This repository contains the implementation of a biologically inspired deep Hebbian (BDH) model designed to evaluate global consistency in long-form narratives. The model addresses the limitations of contemporary large language models (LLMs) in maintaining persistent internal states over extended contexts, such as novels exceeding 100,000 words. By drawing from biological principles, including Hebbian learning and incremental belief updates, the architecture enables adaptive memory formation to assess whether a hypothetical backstory aligns causally and logically with the narrative's evolving constraints.

The core task involves a binary judgment: determining if a proposed backstory for a central character is compatible with the full narrative, focusing on global causal structures rather than local contradictions.

## Problem Statement

Large language models excel in localized tasks like question answering but falter in ensuring global consistency in long-form narratives. Early events impose constraints on later developments, including character evolution, causal pathways, and story-world rules. This project evaluates backstory compatibility by integrating evidence across the entire text, requiring persistent representations that evolve incrementally.

## BDH Concept

### Biological Motivation and Persistent Internal State

Inspired by biological cognition, the model maintains evolving internal representations that accumulate experience over time, contrasting with static LLM contexts. This persistent state preserves beliefs, commitments, and constraints, essential for long-range narrative reasoning.

### Hebbian Learning and Incremental Belief Updates

Hebbian principles strengthen associations through co-activation and weaken them via contradictions. Applied to narratives, this facilitates gradual belief revision, aligning with how humans process stories sequentially.

## Model Architecture

The architecture separates semantic representation from adaptive memory:

- **Input Processing**: Sentences are encoded using the `all-MiniLM-L6-v2` sentence transformer for fixed-dimensional embeddings.
- **Plastic Memory Layers**: A stack of layers combines static parameters (shared priors) with adaptive plastic states that update sequentially during the "read" phase (narrative ingestion) and condition the "query" phase (backstory evaluation).
- **Phases**: 
  - Read: Builds persistent memory from narrative.
  - Query: Propagates backstory embedding through the memory to yield a consistency score.

Stacking layers enables hierarchical abstraction, with deeper models capturing more complex associations.

### Baseline Model

A Mistral-based LLM processes narrative chunks independently, aggregating scores without persistent memory.

### Hybrid Architecture

Augments Mistral by replacing its final layer with a BDH plastic mechanism, combining pretrained representations with adaptive memory for narrative-specific judgments.

## Pretraining

All BDH variants are pretrained on the LiteraryQA dataset (from Hugging Face: `sapienzanlp/LiteraryQA`) to ground semantics, ensuring focus on memory capacity in evaluations.

## Experiment Design

Experiments isolate memory capacity by varying plastic layer depth (1–3 layers) under identical conditions. Hypothesis: Increased depth improves consistency judgments by enhancing constraint reconciliation.

## Training and Evaluation Protocol

- **Training**: Similarity-based objective on shared embeddings.
- **Inference**: Sequential narrative processing builds memory; backstory yields a scalar score for binary classification (consistent/contradict).
- **Metrics**: Accuracy and F1 score.

## Results

Performance trends with increasing layers:

| Model Variant | Accuracy | F1 Score |
|---------------|----------|----------|
| 1 Layer      | 65.00%  | 0.7544  |
| 2 Layers     | 67.50%  | 0.7833  |
| 3 Layers     | 66.25%  | 0.7568  |

Gains peak at two layers, with diminishing returns thereafter due to overparameterization.

## Final Ensemble Model

Combines BDH variants (1–3 layers), Mistral baseline, and hybrid LLM-BDH. Scores are weighted by validation performance and aggregated for robust binary decisions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bdh-plastic-memory.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Includes `sentence-transformers`, `torch`, and others.)

## Usage

1. Prepare narrative and backstory texts.
2. Run inference:
   ```
   python main.py --narrative path/to/novel.txt --backstory path/to/backstory.txt
   ```
   Output: "consistent" or "contradict".

For training on LiteraryQA:
   ```
   python train.py --dataset sapienzanlp/LiteraryQA --layers 2
   ```

## Future Work

- Scale BDH models from scratch with more resources.
- Integrate plasticity into internal transformer components.
- Explore multi-scale plasticity for varied dependencies.
- Extend to lifelong memory across documents.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Trained on LiteraryQA dataset. Inference evaluated on organizer-provided data.
