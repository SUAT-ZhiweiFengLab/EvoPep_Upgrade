# EvoPep: An AI-Driven Design Framework for Multi-Target GLP-1R Agonists in Alzheimer's Disease

**EvoPep** is a generative computational framework based on evolutionary algorithms and deep graph neural networks. It is specifically designed for the de novo discovery and optimization of novel cyclic peptide agonists with polypharmacological profiles for the treatment of Alzheimer's Disease (AD).

## 1. Algorithmic Origins: From EpiMII to EvoPep

The core architectural logic of EvoPep is deeply inspired by **EpiMII** (Integrating Structure and Graph Neural Networks), with significant adaptations and optimizations specifically tailored for cyclic peptide drug discovery:

* **Inheritance**:
    * **Topology-Aware MPNN (Message Passing Neural Network)**: EvoPep inherits EpiMII's core concept of using message passing to process interactions between protein residues. By abstracting the chemical bonds and spatial constraints of cyclic peptides into a graph topology, the MPNN backbone effectively captures long-range structural dependencies without relying on real-time 3D coordinate generation.

* **Optimization & Upgrades**:
    * **Hybrid Generative Architecture (MPNN + CNN + Attention)**: Compared to EpiMII's computationally heavy 3D structure recovery process, EvoPep integrates a **CNN** to rapidly extract local conserved motifs and **Transformer Attention** mechanisms to capture global spatial correlations post-folding. This dramatically improves sequence generation efficiency during the evolutionary search.
    * **Multi-Objective Conformal QSAR Scoring**: EvoPep expands beyond EpiMII's single probability scoring by incorporating a Conformal Prediction-based QSAR model. It simultaneously evaluates binding affinities (pKd) for the primary target (GLP-1R) and 6 critical AD-related targets (e.g., HDAC6, GRIN2A, PI3K), alongside prediction uncertainty intervals.

## 2. Core Architecture

The EvoPep hybrid neural network generator consists of three synergistic modules:
1. **MPNN Backbone**: Simulates the topological graph of cyclic peptides, learning the physicochemical interactions of residues under structural constraints.
2. **1D Convolution (CNN)**: Applies sliding window scanning across the amino acid sequence to rapidly localize key bioactive motifs.
3. **Transformer Attention Heads**: Calculates attention weights between any two residues in the sequence, simulating non-covalent spatial interactions in the folded state.

## 3. Workflow

1. **Initial Population**: The pipeline is seeded with natural GLP-1 sequences or known cyclic peptide analogs (e.g., Liraglutide, Semaglutide scaffolds).
2. **Neural Network-Guided Evolution**: Instead of purely random mutations, EvoPep utilizes the pre-trained hybrid model to compute the logits distribution for each residue position, performing **directional probabilistic sampling** to generate high-quality variants.
3. **Multi-Metric Evaluation**:
    * Computation of primary target affinity (GLP-1R).
    * Evaluation of synergistic neuroprotective engagement across secondary AD targets.
    * Filtering of high-uncertainty samples via Conformal Prediction intervals.
4. **Cyclic Iteration**: Elite sequences are selected to seed the next generation until optimal candidate sequences converge.

## 4. Installation & Quick Start

### Dependencies
Ensure you have `PyTorch` and `NumPy` installed in your environment:
```bash
pip install -r requirements.txt