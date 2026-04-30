# DAT255 Deep Learning — Group 7

Land cover classification using Sentinel-2 satellite imagery and convolutional neural networks.

**Deployed app:** https://huggingface.co/spaces/lasseu/deep-learning-eurosat  
**Experiment tracking:** https://wandb.ai (project: eurosat-cnn)

---

## Project overview

This project investigates land cover classification using the EuroSAT dataset. We train and compare CNNs on RGB images (3 channels) and full 13-band multispectral Sentinel-2 inputs to evaluate performance differences. A series of band ablation experiments is conducted to analyse the contribution of individual spectral bands to classification performance. A Gradio web application is included for interactive demonstration.

---

## Dataset

EuroSAT Sentinel-2 land cover classification dataset: https://zenodo.org/records/7711810

27,000 labeled 64x64 satellite images across 10 land cover classes, available in both RGB (.jpg) and 13-channel multispectral (.tif) formats.

Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Download and unzip both splits into the `data/` folder:

```
data/
    EuroSAT_RGB/
        AnnualCrop/
        Forest/
        ...
    EuroSAT_MS/
        AnnualCrop/
        Forest/
        ...
```

---

## Project structure

```
deep-learning-eurosat/
├── src/
│   ├── dataset.py          data loading for RGB and multispectral formats
│   ├── model.py            CNN architecture with configurable input channels
│   ├── train.py            training loop with W&B logging
│   └── evaluate.py         confusion matrix and per-class metrics
├── scripts/
│   ├── save_val_split.py   saves validation image paths to val_images.txt
│   ├── make_canvas.py      stitches validation images into demo canvases
│   └── plot_models.py      plots model architecture diagrams
├── notebooks/              exploratory data analysis, scripts used for training on Colab
├── demo/
│   └── canvases/           pre-generated canvas images from the validation set for the web app
├── results/
│   └── figures/            confusion matrices and evaluation plots
├── app.py                  Gradio web application
├── sweep.yaml              W&B hyperparameter sweep configuration
└── requirements.txt        Not necessary if running inside Docker, but used for Hugging Face deployment
```

---

## Setup

The project runs inside a Dev Container. Open the repository in VS Code and select "Reopen in Container" when prompted.


Log in to Weights & Biases:

```bash
wandb login
```

---

## Training

All training runs are logged to Weights & Biases automatically.

Train on RGB images:

```bash
python src/train.py --mode rgb
```

Train on all 13 multispectral bands (requires Colab or GPU):

```bash
python src/train.py --mode ms
```

Train on a subset for local testing:

```bash
python src/train.py --mode ms --max_per_class 500
```

Band ablation experiment (drop NIR bands):

```bash
python src/train.py --mode ms --band_indices 0 1 2 3 4 5 6 9 10 11 12
```

Available arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `rgb` | Dataset mode: `rgb` or `ms` |
| `--model_version` | `batchnorm_3conv` | Architecture variant |
| `--band_indices` | all 13 | MS band indices to use (0-12) |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--dropout` | 0.4 | Dropout rate |
| `--max_per_class` | None | Max images per class (None = full dataset) |

Available model versions: `baseline`, `batchnorm`, `batchnorm_3conv`, `batchnorm_3conv_gap`

---

## Evaluation

Change the variables to match the checkpoint, mode and band_indices and then run evaluation.

```bash
python src/evaluate.py
```

Outputs a confusion matrix and per-class precision, recall and F1 score. Results are saved to `results/`.

---

## Hyperparameter sweep

```bash
wandb sweep sweep.yaml
wandb agent your-username/eurosat-cnn/SWEEP_ID
```

The sweep searches over learning rate, dropout rate, and batch size using Bayesian optimisation. Use `--max_per_class 500` in the sweep config for manageable run times.

---

## Web application

The Gradio app classifies satellite images by dividing them into 64x64 tiles and predicting the land cover class for each tile.

Run locally:

```bash
python app.py
```

Generate demo canvases from the validation split before running the app:

```bash
python scripts/save_val_split.py
python scripts/make_canvas.py
```

The app is deployed at: https://huggingface.co/spaces/lasseu/deep-learning-eurosat

---

## Sentinel-2 band reference

| Index | Band | Description |
|-------|------|-------------|
| 0 | B01 | Coastal aerosol |
| 1 | B02 | Blue |
| 2 | B03 | Green |
| 3 | B04 | Red |
| 4 | B05 | Red Edge 1 |
| 5 | B06 | Red Edge 2 |
| 6 | B07 | Red Edge 3 |
| 7 | B08 | NIR (Near Infrared) |
| 8 | B08A | Narrow NIR |
| 9 | B09 | Water vapour |
| 10 | B10 | SWIR cirrus |
| 11 | B11 | SWIR 1 |
| 12 | B12 | SWIR 2 |