# Vulnerability Dataset

This repository contains a dataset for vulnerability detection used in our research study:  
**A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code**.

## 📂 Dataset Overview

The dataset is derived from **publicly available datasets** used in prior research on vulnerability detection:
- **Devign:** _Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks_ ([DOI: 10.48550/arXiv.1909.03496](https://doi.org/10.48550/arXiv.1909.03496))
- **REVEAL:** _Deep Learning based Vulnerability Detection: Are We There Yet?_ ([DOI: 10.48550/arXiv.2009.07235](https://doi.org/10.48550/arXiv.2009.07235))

To facilitate ease of use, we have **preprocessed and reformatted** the datasets into a single `combined.pkl` file containing labeled source code functions.

### 🔹 **File**
| File Name       | Description |
|----------------|------------|
| `combined.pkl` | A dictionary containing vulnerability-labeled functions from multiple projects (FFmpeg, Chrome, Debian, Qemu). |

### 🔹 **Structure**
`combined.pkl` contains a dictionary where:
- `combined_data["FFmpeg"]` 
- `combined_data["Chrome"]` 
- `combined_data["Debian"]` 
- `combined_data["Qemu"]` 

Each dataset consists of **source code functions** and their **binary labels** indicating whether they are **vulnerable (1) or non-vulnerable (0)**.

## 📖 Usage

To load the dataset in Python:
```python
import pickle

with open("combined.pkl", "rb") as f:
    combined_data = pickle.load(f)

# Access datasets
ffmpeg_data = combined_data["FFmpeg"]
chrome_data = combined_data["Chrome"]
debian_data = combined_data["Debian"]
qemu_data = combined_data["Qemu"]

print(ffmpeg_data)
