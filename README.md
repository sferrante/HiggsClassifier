# HiggsClassifier

Small ML experiments on a **physics-motivated jet classification** task: a simulated **Z → b b̄** final state generated with **MadGraph**. This repo is mainly a playground for:
- **Supervised classification** (signal vs background)
- **Unsupervised / weakly-supervised ideas** (CWoLa-style separation)
- Two broader ML phenomena:
  - **Double descent**
  - **Neural scaling laws** + **compute-optimal** behavior

---

## Table of Contents
- [Contents](#contents)
- [Background](#background)
- [Setup](#setup)
- [Experiments](#experiments)
  - [Supervised classifier](#supervised-classifier)
  - [Double descent](#double-descent)
  - [Scaling laws](#scaling-laws)
  - [CWoLa-style unsupervised separation](#cwola-style-unsupervised-separation)
- [Results](#results)
- [Compute-optimal scaling plot](#compute-optimal-scaling-plot)
- [Notes](#notes)

---

## Contents

> **Note:** rename these bullets to match your actual filenames (I’m using clean, descriptive defaults).

- `notebooks/01_supervised_classifier.ipynb`  
  Train a neural classifier for signal/background discrimination; includes basic metrics and sanity checks.

- `notebooks/02_double_descent.ipynb`  
  Explore **double descent** by varying training duration / effective fitting regime and tracking test error.

- `notebooks/03_scaling_laws.ipynb`  
  Sweep model size (parameter count) and observe approximate **power-law** behavior of loss vs scale.

- `notebooks/04_cwola_autoencoder.ipynb`  
  An autoencoder-driven / weakly-supervised workflow for **CWoLa-style** separation without per-event labels.

- `scripts/plot_compute_optimal.py`  
  Utility script to generate the compute-optimal scaling plot shown at the end of this README.

- `results/`  
  Saved sweep outputs (CSV/JSON) used to reproduce plots without rerunning training.

- `assets/`  
  Figures displayed in the README.

---

## Background

In collider physics, many searches reduce to learning subtle differences between event classes.
This repo uses a toy-but-realistic setup: simulated **Z → b b̄** events (plus background),
converted into a fixed set of kinematic features and used for classification.

The goal here is not “state of the art,” but **understanding modern ML behavior** (generalization curves,
double descent, scaling trends) in a concrete scientific dataset.

---

## Setup

### Environment
Recommended: Python 3.9+.

```bash
# optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
