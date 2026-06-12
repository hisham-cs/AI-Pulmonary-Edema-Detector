# Pulmonary Edema Detection from Chest X-ray Images Using Deep Learning

A deep learning-based web application that detects **Pulmonary Edema** from chest X-ray images. The app combines four convolutional neural networks through soft-voting ensemble and explains its predictions with Grad-CAM heatmaps, all through an interactive Streamlit interface.

---

## 📌 Project Overview

Pulmonary edema is a serious condition in which fluid accumulates in the lungs, and chest X-rays are a primary tool for detecting it. This project frames the problem as a **binary classification task** (Normal vs. Edema). Instead of relying on a single network, the application averages the predictions of four trained models to produce a more reliable final decision, and visualizes the image regions that influenced each prediction.

## ✨ Features

All features below are implemented in `app.py`:

- **Chest X-ray upload** — accepts `.jpg`, `.jpeg`, and `.png` images.
- **Input validation (optional)** — uses the OpenAI API (GPT-4o Vision) to verify the uploaded image is a real chest X-ray before classification, with a "Continue Anyway" override. Skipped automatically if no API key is configured.
- **4-model soft-voting ensemble** — DenseNet121 (TorchXRayVision), ResNet50, EfficientNet-B0, and ConvNeXt-Tiny each predict independently; their edema probabilities are averaged into the final diagnosis.
- **Prediction confidence** — displays the ensemble edema probability and each model's individual vote with a probability bar.
- **Model agreement indicator** — reports High / Medium / Low agreement based on how much the four models' probabilities differ, with a caution warning on low agreement.
- **CLAHE preprocessing** — contrast-limited adaptive histogram equalization is applied to improve X-ray contrast before inference.
- **Grad-CAM explainability** — view a per-model heatmap for any of the four networks, or a fused "Ensemble CAM" that averages all four attention maps.

## 🛠 Tech Stack

Based on `requirements.txt`:

| Category | Libraries |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision, TorchXRayVision, timm |
| Explainability | grad-cam (Grad-CAM) |
| Image Processing | OpenCV, Pillow, Albumentations |
| Data & Utilities | NumPy, Pandas, scikit-learn, Matplotlib, tqdm |
| Web App | Streamlit |
| Image Validation | OpenAI API, python-dotenv |

## 📂 Project Structure

```
AI-Pulmonary-Edema-Detector/
├── app.py                          # Streamlit application (validation, ensemble inference, Grad-CAM)
├── requirements.txt                # Python dependencies
├── model/                          # Trained model weights
│   ├── densenet121_weights.pth
│   ├── resnet50_weights.pth
│   ├── efficientnet_weights.pth
│   └── convnext_weights.pth
└── .gitattributes
```

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/hisham-cs/AI-Pulmonary-Edema-Detector.git
cd AI-Pulmonary-Edema-Detector
```

2. **Create and activate a virtual environment (Windows)**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **(Optional) Enable chest X-ray validation**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

If the key is missing, the app skips validation and proceeds directly to classification.

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload a chest X-ray image, and the app will:

1. Validate that the image is a chest X-ray (if an OpenAI API key is configured).
2. Run all four models and display the final ensemble diagnosis (**NORMAL** or **EDEMA**) with its probability.
3. Show each model's individual prediction and confidence in the sidebar.
4. Let you switch between per-model and ensemble Grad-CAM heatmap views.

## 🧠 Model Files

Trained weights for the four networks are stored in the `model/` folder:

| File | Architecture |
|---|---|
| `densenet121_weights.pth` | DenseNet121 (TorchXRayVision, pretrained on chest X-rays) |
| `resnet50_weights.pth` | ResNet50 |
| `efficientnet_weights.pth` | EfficientNet-B0 |
| `convnext_weights.pth` | ConvNeXt-Tiny |

Each model was fine-tuned for 2-class output (Normal / Edema) and is loaded automatically by `app.py` at startup.

## 🗂 Dataset

The training dataset is **not included** in this repository. The application performs inference only — it requires a single chest X-ray image uploaded through the web interface, not a dataset.

## ⚠️ Limitations

- This project is for **educational and research purposes only**.
- It is **not a replacement for professional medical diagnosis**. Always consult a qualified radiologist or physician.
- Model performance depends on the quality and diversity of the data used during training, and may not generalize to X-rays from different hospitals, devices, or populations.
- The optional image validation step requires an OpenAI API key and an internet connection.
- Training code and evaluation results are not included in this repository.

## 🔮 Future Improvements

- Publish evaluation results (accuracy, sensitivity, specificity, ROC-AUC, confusion matrices)
- Add training notebooks or scripts
- Add dataset documentation
- Replace the OpenAI-based validator with a local (offline) chest X-ray validation model
- Improve the Streamlit UI and add batch prediction
- Deploy the application to a cloud platform

## 👥 Contributors

* **Hisham Almalki**
* **Ali Almufarriji**
* **Saleh Alsulami**


## 📄 License

This project is for academic and educational purposes.
