# HiggsClassifier

Small ML experiments on a **physics-motivated jet classification** task: a simulated **Z → b b̄** final state generated with **MadGraph**. Contains:
- **Supervised classification** (signal vs background)
- **Unsupervised / weakly-supervised ideas** (CWoLa-style separation)
- Two broader ML phenomena:
  - **Neural scaling laws** + **compute-optimal** behavior
  - **Double descent**

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

- `HiggstrahlungClassifier_Supervised.ipynb`  
  Trains a neural classifier for signal/background discrimination; explores scaling laws, and calculates the **compute-optimal** scaling law. 

- `HiggstrahlungClassifier_Unsupervised.ipynb`  
  An autoencoder-driven / unsupervised workflow for **CWoLa-style** separation without per-event labels.

- `HiggstrahlungClassifier.py`
 Defines the MLP and Autoencoder models, training procedures, plotting and more.

- `Datasets/ee_Zbb_noH.lhe`  
  Background Dataset in LHE format.
  
- `Datasets/ee_ZH_Zbb.lhe`  
  Signal Dataset in LHE format.


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
