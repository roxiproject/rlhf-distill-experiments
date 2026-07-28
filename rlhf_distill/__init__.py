"""rlhf_distill: a numpy-only toolkit for knowledge distillation experiments.

Implements a small MLP with a manually-derived backward pass, an Adam
optimizer from scratch, a toy 2D classification task, and a numerically
stable temperature-scaled KL distillation loss combined with a hard-label
cross-entropy term.
"""

__version__ = "0.1.0"
