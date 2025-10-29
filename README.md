# Zero-Shot Cross-Project Vulnerability Detection
**A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code**  
DOI: https://doi.org/10.1007/s10664-025-10749-4

---

## Setup

```bash
git clone https://github.com/Radowan98/ZSVulD.git
cd ZSVulD
pip install -r requirements.txt
```

Ensure the `dataset/` folder contains:
- `combined_data.zip` (and/or `combined_data.pkl` if you plan to regenerate embeddings)
- `qemu_embeddings.npy`, `qemu_labels.npy`
- `ffm_embeddings.npy`, `ffm_labels.npy`
- `deb_embeddings.npy`, `deb_labels.npy`
- `chr_embeddings.npy`, `chr_labels.npy`

---

## Generate embeddings (optional)

If you prefer to create the `.npy` files yourself from `combined_data.pkl` using CodeBERT:

```bash
python src/generate_embeddings_codebert.py
```

This reads `dataset/combined_data.pkl` and writes `<project>_embeddings.npy` and `<project>_labels.npy` to `dataset/`.

---

## Run experiments

Setting 1: Devign (Qemu + FFmpeg) → ReVeal (Debian + Chrome)
```bash
python src/zero_shot_model.py --setting 1
```

Setting 2: Qemu + ReVeal (Debian + Chrome) → FFmpeg
```bash
python src/zero_shot_model.py --setting 2
```

Setting 3: Devign (Qemu + FFmpeg) + Chrome → Debian
```bash
python src/zero_shot_model.py --setting 3
```

---

## Citation

```
@article{Haque2026ZeroShotVulnDetection,
  author    = {Haque, Radowanul and Ali, Ahsan and McClean, Sally and others},
  title     = {A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code},
  journal   = {Empirical Software Engineering},
  volume    = {31},
  number    = {3},
  year      = {2026},
  doi       = {10.1007/s10664-025-10749-4},
  url       = {https://doi.org/10.1007/s10664-025-10749-4}
}

```


