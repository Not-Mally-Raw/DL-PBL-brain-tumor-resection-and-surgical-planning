# Phase 1–6 Notebooks (DL + RL)

This repo packages the **six phase notebooks** for a brain-tumor surgical planning pipeline that combines **deep learning** (segmentation + safety mapping) with **reinforcement learning** (trajectory optimization).

## Contents

Run the notebooks in order:

1. `phase-1-dl-2.ipynb` — Phase 1: data harmonization / preprocessing
2. `phase-2-brats-pretraining-dl-3.ipynb` — Phase 2: segmentation backbone training (BraTS-style)
3. `phase-3-transfer-learning-for-deep-learning (2).ipynb` — Phase 3: transfer learning (safety head)
4. `phase-4-preoperative-surgical-context.ipynb` — Phase 4: pre-operative surgical context (EDT + safety gradients)
5. `Phase5_Kaggle_Training_Backend.ipynb` — Phase 5: RL MDP environment + training backend (Kaggle-oriented)
6. `Phase6_Colab_Frontend.ipynb` — Phase 6: RL/Q-learning execution + frontend workflow (Colab-oriented)

## How to run

- **Locally (recommended for Phases 1–4):** open in Jupyter Lab/Notebook and execute top-to-bottom.
- **Cloud GPU (recommended for Phases 5–6):** these notebooks are written for Kaggle/Colab-style environments.

Typical local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter
jupyter lab
```

Then open each notebook and follow any in-notebook install/data instructions.

## Repo goal

Keep the phase deliverables **versioned and shareable** (single place for Phase 1 → Phase 6) so the full workflow can be reviewed, re-run, and iterated.
