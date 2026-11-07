---
title: NeuroScan AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# NeuroVision

NeuroVision (also presented as **NeuroScan AI**) is a research-oriented web application for exploring brain MRI scans with deep-learning-based tumor classification and segmentation. It combines a Flask interface with TensorFlow models to display a prediction and a segmentation mask for an uploaded image.

> **Medical disclaimer:** This project is for research and educational use only. It is not a medical device and must not be used to diagnose, treat, or make decisions about a patient. Have all medical images interpreted by a qualified healthcare professional.

## What it does

- Upload an MRI image through a responsive web interface.
- Run classification and segmentation inference with the supplied model architectures and weights.
- View the predicted mask and analysis results.
- Optionally create an account and save scan history.
- Use MongoDB for persistent history and Cloudinary for image storage.

The project also includes scripts and a notebook for dataset inspection and model-comparison experiments. See [SETUP.md](./SETUP.md) for additional setup, troubleshooting, and testing notes.

## Requirements

- Python 3.11 (also used by the included Docker image)
- pip
- The model architecture and checkpoint files listed below
- Cloudinary credentials (required by the current application startup configuration)
- MongoDB is optional; without `MONGO_URI`, the app uses local JSON storage

## Run locally

From the repository root:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux (use this instead of the Windows activation command)
# source .venv/bin/activate

python -m pip install -r requirements-web.txt
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

On Windows Command Prompt, use `copy .env.example .env`. Set the following values in `.env`:

```dotenv
FLASK_SECRET_KEY=replace-with-a-long-random-value
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

`MONGO_URI` is optional. Do not commit `.env` or publish its credentials.

Make sure these model files are present in the repository root:

| File | Purpose |
| --- | --- |
| `resnet-50-MRI.json` | Classification model architecture |
| `weights.hdf5` | Classification model weights |
| `ResUNet-MRI.json` | Segmentation model architecture |
| `weights_seg.hdf5` | Segmentation model weights |

Model checkpoints are excluded from Git by `.gitignore`; they may need to be obtained separately. The TCGA dataset folders are used for dataset and experiment workflows, but are not required to run the web application.

Start the app:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). To configure the bind address or port, set `FLASK_HOST` or `FLASK_PORT` in `.env`.

## Docker / Hugging Face Spaces

The included [Dockerfile](./Dockerfile) runs the web application on port `7860`. Build and start it with:

```bash
docker build -t neurovision .
docker run --env-file .env -p 7860:7860 neurovision
```

The Docker image requires both model checkpoints at build time. For a Hugging Face Docker Space, configure the required credentials as Space secrets and provide the weights using an appropriate large-file storage method. Do not include `.env` or TCGA data in a public deployment. See [SETUP.md](./SETUP.md) for deployment details.

## Data and experiments

The research dataset is based on the TCGA lower-grade glioma (LGG) collection of brain MRI images and FLAIR abnormality masks. Dataset metadata is provided in `data.csv` and `data_mask.csv`. See [SETUP.md](./SETUP.md) for dataset placement and experiment instructions.

## API

The Flask application includes endpoints for health checks, prediction, configuration, authentication, and scan history. The primary prediction endpoint is `POST /api/predict`; the health endpoint is `GET /api/health`. Start the app locally and use the browser's network tools or inspect `app.py` for request and response details.

## Project resources

- [Setup and troubleshooting](./SETUP.md)
- [Contribution guide](./CONTRIBUTING.md)
- [License](./LICENSE)

## Contributing

Bug fixes, documentation improvements, and reproducible research contributions are welcome. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a pull request.