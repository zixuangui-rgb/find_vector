# find_vector

This repository contains the pre-training-stage validation experiment for finding and testing two candidate representation directions in a Qwen3.5-style instruction model:

- emotional validation
- belief endorsement

The current scope is limited to direction discovery and validation before any post-training.

## Current Contents

- `docs/01_concept_rubric.md`: concept definitions and labeling rubric
- `docs/02_pretraining_vector_discovery_experiment_plan.md`: detailed experimental plan
- `data/2x2_items_all_v1.jsonl`: v1 controlled 2x2 dataset
- `data/2x2_items_train_v1.jsonl`, `data/2x2_items_dev_v1.jsonl`, `data/2x2_items_test_v1.jsonl`: scenario-level splits
- `review/`: review notes, dataset summary, audit notes, and user-facing sample
- `scripts/generate_2x2_dataset.py`: reproducible v1 dataset generator

Large server artifacts such as activations, vectors, checkpoints, and raw generation logs are intentionally ignored unless explicitly curated.

