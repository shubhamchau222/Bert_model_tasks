
```bash
# Create a new environment (optional)
conda create -n myenv python=3.9 -y

# Activate the environment
conda activate myenv
```

```bash
# For CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# for GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

```bash
pip install transformers

#if you prefare using conda for transformers
conda install -c conda-forge transformers
```

```python
# Check PyTorch
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Check Transformers
from transformers import pipeline
print("Transformers installed successfully!")

```
```bash
conda install ipykernel
```

```txt
# sample template for req.txt

torch==<specific-version>
torchvision==<specific-version>
torchaudio==<specific-version>
transformers==<specific-version>
ipykernel==<specific-version>

```
```python
# check versions of install libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("NumPy version:", np.__version__)
print("Seaborn version:", sns.__version__)
```
