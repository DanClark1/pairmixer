# ============================================================
# CELL 1: CONFIGURATION
# ============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Import libraries
import boltz

try:
    import pairmixer
    PAIRMIXER_AVAILABLE = True
    print("✓ Pairmixer library loaded")
except ImportError:
    PAIRMIXER_AVAILABLE = False
    print("⚠ Pairmixer library not found - will skip Pairmixer models")

# Configuration
EXAMPLES = {
    'prot': '/content/drive/MyDrive/gdl/examples/prot.yaml',
    'ligand': '/content/drive/MyDrive/gdl/examples/ligand.yaml',
    'prot_no_msa': '/content/drive/MyDrive/gdl/examples/prot_no_msa.yaml',
    'multimer': '/content/drive/MyDrive/gdl/examples/multimer.yaml',
}

MODELS = {
    'boltz1': {'library': boltz, 'model_id': 'boltz1'},
}

# Add Pairmixer only if available
if PAIRMIXER_AVAILABLE:
    MODELS['pairmixer'] = {'library': pairmixer, 'model_id': 'pairmixer'}

DRIVE_PATH = Path('/content/drive/MyDrive/boltz_analysis')
DRIVE_PATH.mkdir(exist_ok=True)

# Analysis parameters
NUM_SWAPS = 20
NUM_BLOCKS = 16
TEMPERATURE = 0.1

print(f"Will analyze {len(EXAMPLES)} examples × {len(MODELS)} models")
print(f"Models configured: {list(MODELS.keys())}")
print(f"Saving to: {DRIVE_PATH}")