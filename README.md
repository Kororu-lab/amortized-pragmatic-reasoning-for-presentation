# Amortized Pragmatic Reasoning

This repository implements an amortized Rational Speech Acts (RSA) model for pragmatic language generation.

In amortized RSA, nested pragmatic reasoning is learned by a neural network so that test-time generation is fast and informative. This repo provides end-to-end scripts to generate/prepare data (ShapeWorld, Colors-in-Context), train literal listeners/speakers and amortized speakers (with optional length penalties), and evaluate with standard metrics and compositional analyses.

Paper: [Learning to refer informatively by amortizing pragmatic reasoning](https://arxiv.org/abs/2006.00418) | PDF: [arXiv:2006.00418](https://arxiv.org/pdf/2006.00418)

**ShapeWorld**: A reference game dataset with simple geometric shapes varying in shape (circle, square, rectangle, ellipse) and color (red, blue, green, yellow, white, gray).

**Colors in Context**: A reference game datasets with Human-generated reference game dataset with referring expressions for colors in context. Download from [Monroe et al., 2017](https://cocolab.stanford.edu/datasets/colors.html).

**Literal Listener (L0)**: Selects the target referent given an utterance; serves as the base listener component.

**Literal Speaker (S0)**: Generates utterances directly from data without explicit pragmatic reasoning.

**Amortized Speaker**: Learns a neural approximation to pragmatic reasoning for fast, informative generation; supports length penalty via `--penalty length` with weight `--lmbd`.

**REINFORCE Speaker**: Variant of amortized speaker using policy gradients (set `activation: multinomial` in config).

---

## Quickstart

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train an amortized speaker (data auto-generated from config)
python scripts/train.py --config configs/shapeworld_amortized.yaml

# 3) Evaluate on the test split
python scripts/evaluate.py --config configs/shapeworld_amortized.yaml --split test
```

---

## Usage

### Training

Train models using YAML config files. Data and dependencies (vocab, literal listeners) are automatically generated if missing.

```bash
# ShapeWorld
python scripts/train.py --config configs/shapeworld_s0.yaml          # Literal speaker
python scripts/train.py --config configs/shapeworld_amortized.yaml   # Amortized (auto-trains L0)
python scripts/train.py --config configs/shapeworld_reinforce.yaml   # REINFORCE (auto-trains L0)

# Colors in Context (download dataset first - see above)
python scripts/train.py --config configs/colors_s0.yaml              # Literal speaker
python scripts/train.py --config configs/colors_amortized.yaml       # Amortized (auto-trains L0)
python scripts/train.py --config configs/colors_reinforce.yaml       # REINFORCE (auto-trains L0)
```

**Config file structure** (see `configs/` for examples):

```yaml
# Dataset configuration
dataset:
  name: shapeworld               # or colors
  n_examples: 1000               # number of examples per file
  n_images: 3                    # images per scene (target + distractors)
  data_type: reference           # task type
  img_type: single               # image layout
  generalization: null           # or: new_color, new_shape, new_combo

# Training configuration
training:
  model_type: amortized          # l0, s0, amortized
  batch_size: 32
  epochs: 100
  learning_rate: null            # auto-set based on model type
  cuda: true
  debug: false
  seed: 42                       # random seed for reproducibility
  
  # Amortized speaker specific
  activation: gumbel             # gumbel, softmax, multinomial
  penalty: length
  lmbd: 0.01                     # length penalty weight
  tau: 1.0                       # temperature for Gumbel-Softmax
  
  # Literal listener (L0) pretraining config (auto-triggered if needed)
  l0_epochs: 100
  l0_learning_rate: 0.0001
  l0_batch_size: 32

# Model architecture
model:
  embedding_dim: 256
  hidden_size: 256
  vision_model: Conv4
  max_sequence_length: 10
```

**Command-line overrides**: You can override any config value via CLI:
```bash
python scripts/train.py --config configs/shapeworld_amortized.yaml --epochs 200 --cuda
```

### Evaluate

Evaluate trained models using the same config files:

```bash
# Evaluate using config (automatically finds trained models)
python scripts/evaluate.py --config configs/shapeworld_amortized.yaml --split test

# Or manually specify paths
python scripts/evaluate.py --config configs/shapeworld_s0.yaml --split val --speaker_path experiments/saved_models/shapeworld/literal_speaker.pt
```

**Arguments**:
- `--config`: path to config file (same as training)
- `--split`: `train`, `val`, or `test` (default: `test`)
- `--speaker_path`: optional, override model path
- `--listener_path`: optional, override listener path

---

## Citation

You may cite this work as follows:

- [Learning to refer informatively by amortizing pragmatic reasoning](https://arxiv.org/abs/2006.00418) ([PDF](https://arxiv.org/pdf/2006.00418))

```bibtex
@misc{white2020learningreferinformativelyamortizing,
      title={Learning to refer informatively by amortizing pragmatic reasoning}, 
      author={Julia White and Jesse Mu and Noah D. Goodman},
      year={2020},
      eprint={2006.00418},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2006.00418}, 
}
```