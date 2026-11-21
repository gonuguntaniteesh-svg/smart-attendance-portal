# Smart Attendance Portal Using Face Recognition
### Streamlit Production Deployment & Research Benchmark Submission

---

## 📌 1. Project Overview & Faculty Mandate Compliance

This repository implements an automated, biometric attendance system powered by deep learning and machine learning.
The implementation strictly follows the faculty instructions:

> **"For deployment use your best methods pipeline, not all the combinations."**
> **"Upload the necessary project requirements. Upload the `.ipy` / `.ipynb` file. In the second section, upload a screenshot of the deployed dashboard. Deploy using Streamlit. Use only the best deployment method, not multiple combinations."**

### Selected Best Deployment Pipeline: **Vision Transformer (ViT) + SVM**
- **Feature Extractor**: Vision Transformer (`google/vit-base-patch16-224-in21k`, 768-d CLS token representation)
- **Classifier**: Support Vector Machine (`StandardScaler` + `SVC(kernel="rbf", C=1.0, probability=True)`)
- **Macro F1-Score**: **89.88%**
- **Overall Accuracy**: **90.00%**
- **Macro Precision**: **91.50%**
- **Macro Recall**: **90.00%**
- **Deployment Platform**: **Streamlit** (`app.py`)

All other 15 experimental combinations (ResNet50, MobileNetV2, InceptionV3 combined with SVM, XGBoost, AdaBoost, Gradient Boosting) are completely preserved in the research files (`smart_attendance_research.ipynb`, `smart_attendance_research.ipy`, `smart_attendance_portal.py`, and `results/ML_Classification_Results.csv`) and are not loaded into the deployment runtime.

---

## 📁 2. Submission Files Checklist

| File Name | Description | Faculty Requirement |
|---|---|---|
| **`app.py`** | Main Streamlit Application (Dashboard, Webcam Recognition, Student Enrollment, History, Excel Export) | Required Deployment Entry Point |
| **`requirements.txt`** | Minimal, Linux/Cloud compatible dependencies (`opencv-python-headless`, etc.) | Required Project Requirements |
| **`smart_attendance_research.ipynb`** | Full research Jupyter Notebook with 16 pipeline evaluations, confusion matrices, and ROC-AUC | Required Notebook File |
| **`smart_attendance_research.ipy`** | IPython research script export matching exact faculty requirement | Required `.ipy` File |
| **`smart_attendance_portal.py`** | Original preserved Colab research code | Preserved Research Baseline |
| **`results/dashboard_screenshot.png`** | Verified screenshot of the live, working Streamlit dashboard | Required Dashboard Screenshot |
| **`models/`** | Pre-trained deployment artifacts (`vit_svm_classifier.pkl`, `label_encoder.pkl`, `embeddings_store.pkl`, `config.json`) | Deployment Artifacts |
| **`README.md`** | Comprehensive architecture, benchmark analysis, and deployment guide | Documentation |

---

## 📊 3. Research Evaluation: Comparison of All 16 Combinations

| # | Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Status |
|---|---|---|:---:|:---:|:---:|:---:|---|
| 1 | **ResNet50** | **SVM** | 62.50% | 68.45% | 62.50% | 61.57% | *Initial Candidate* |
| 2 | ResNet50 | XGBoost | 65.00% | 60.38% | 65.00% | 61.32% | Research Reference |
| 3 | ResNet50 | AdaBoost | 40.00% | 41.05% | 40.00% | 35.86% | Research Reference |
| 4 | ResNet50 | Gradient Boosting | 50.00% | 41.50% | 50.00% | 44.91% | Research Reference |
| 5 | MobileNetV2 | SVM | 50.00% | 49.33% | 50.00% | 46.90% | Research Reference |
| 6 | MobileNetV2 | XGBoost | 42.50% | 42.00% | 42.50% | 40.57% | Research Reference |
| 7 | MobileNetV2 | AdaBoost | 30.00% | 37.17% | 30.00% | 30.00% | Research Reference |
| 8 | MobileNetV2 | Gradient Boosting | 45.00% | 45.25% | 45.00% | 42.59% | Research Reference |
| 9 | InceptionV3 | SVM | 72.50% | 79.33% | 72.50% | 71.29% | Research Reference |
| 10 | InceptionV3 | XGBoost | 60.00% | 64.33% | 60.00% | 59.21% | Research Reference |
| 11 | InceptionV3 | AdaBoost | 37.50% | 45.14% | 37.50% | 35.22% | Research Reference |
| 12 | InceptionV3 | Gradient Boosting | 52.50% | 58.67% | 52.50% | 51.84% | Research Reference |
| **13** | **ViT (Vision Transformer)** | **SVM** | **90.00%** | **91.50%** | **90.00%** | **89.88%** | 🏆 **DEPLOYED PIPELINE** |
| 14 | ViT (Vision Transformer) | XGBoost | 67.50% | 69.05% | 67.50% | 67.27% | Research Reference |
| 15 | ViT (Vision Transformer) | AdaBoost | 50.00% | 50.62% | 50.00% | 46.39% | Research Reference |
| 16 | ViT (Vision Transformer) | Gradient Boosting | 70.00% | 80.33% | 70.00% | 71.52% | Research Reference |

---

## 🚀 4. How to Run Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically at: `http://localhost:8501`.

---

## ☁️ 5. Deployment to GitHub & Streamlit Community Cloud

1. **Initialize Git Repository and Push**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Smart Attendance Portal using ViT + SVM on Streamlit"
   git remote add origin https://github.com/<your-username>/smart-attendance-portal.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Sign in with GitHub.
   - Click **New app**.
   - Select your repository, branch (`main`), and set **Main file path** to `app.py`.
   - Click **Deploy!**

---

## 🔒 6. Key Application Features

1. **Webcam Biometric Attendance**:
   - Real-time face detection with Haar Cascade.
   - Deep 768-dimensional feature representation extraction using Vision Transformer.
   - **Unknown-Person Rejection**: Rejects any face where SVM predicted confidence is below 65%.
   - **Duplicate Attendance Prevention**: Automatically queries `(student_id, attendance_date, subject)` and blocks duplicate entries on the same day.
2. **Student Registration & Immediate Biometric Recognition**:
   - Register new admissions with ID, Name, Department, Year, Section, Email, and Parent Email.
   - Capture or upload face photos; embeddings are extracted and the SVM classifier is retrained immediately.
3. **Attendance Reports & Export**:
   - Filter by date, department, or subject.
   - One-click formatted Excel (.xlsx) export with auto-formatted column headers and summary metrics.
   - Raw CSV log downloads.
