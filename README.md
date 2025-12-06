# Decision-Transformer-New
A new implementation of Decision Transformer, using Gymnasium and Minari. 

No need to install d4rl and mujoco-py.

Supports Python versions >= 3.11.

# Build Environment
```powershell
pip install -r requirements.txt
```

# Download Datasets
```powershell
cd data
python download_datasets.py
```

# Run Experiments
```powershell
python experiment.py
```

# Results
| Dataset | Environment | DT (this repo) 100k updates |  DT (official) 100k updates |
| - | - | - | - |
| Medium | Hopper | 78.89 $\pm$ 28.36 | 67.60 $\pm$ 01.00 |

