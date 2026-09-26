---
tags:
- recursive-reasoning
- arc-agi
- tiny-models
license: mit
---

# TRM Sudoku Extreme (MLP)

Tiny Recursion Model (7M params) with MLP layers trained on extreme Sudoku puzzles

## Model Details

This is a checkpoint from the TinyRecursiveModels project, implementing recursive reasoning for puzzle solving.

**Checkpoint:** `step_65100`

## Usage

To load this checkpoint:

```python
import torch

# Load the checkpoint
checkpoint = torch.load("model.pt", map_location="cpu")

# The checkpoint contains:
# - 'model': model state dict
# - 'ema_model': EMA model state dict (if EMA was enabled)
# - 'puzzle_emb': puzzle embeddings
# - 'optimizer': optimizer state
# - 'config': training configuration
# - 'step': training step number
# - 'epoch': training epoch number

# Access model weights
model_state = checkpoint['model']
config = checkpoint['config']
```

## Training Info

- **Checkpoint:** step_65100
- **Step:** N/A
- **Epoch:** N/A

## Citation

If you use this model, please cite the TinyRecursiveModels repository.
