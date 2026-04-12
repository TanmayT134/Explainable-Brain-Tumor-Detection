# 🧠 Explainable Deep Learning Model For Brain Tumor Detection and Classification using MRI Images

---

## 🚀 Live Demo

👉 **Try the Application:**  
🔗 https://brain-mri-ai.streamlit.app/

---

## 📌 Overview

This project presents an **AI-powered diagnostic system** that detects and classifies brain tumors from MRI scans using a **Convolutional Neural Network (CNN)**.

Unlike traditional models, this system integrates **Explainable AI (XAI)** using **Grad-CAM**, allowing users to visually understand *why* the model made a prediction.

It is deployed as an **interactive web application** built with Streamlit.

---

## ✨ Features

- 🧠 Brain tumor classification from MRI images  
- 📊 Multiclass prediction (4 tumor categories)  
- 🔥 Grad-CAM visualization for explainability  
- 📈 Confidence score & probability distribution  
- ⚠️ Uncertainty detection  
- 🔍 Image quality assessment  
- 📄 Clinical-style PDF report generation  
- 🌐 Fully deployed web application  

---

## 🧬 Tumor Classes

| Class | Description |
|------|------------|
| Glioma | Tumor in brain/glial cells |
| Meningioma | Tumor in brain membranes |
| Pituitary | Tumor in pituitary gland |
| No Tumor | Normal brain MRI |

---

## 🧠 Model Architecture

- Convolutional Neural Network (CNN)
- Key Layers:
  - Convolution + ReLU  
  - Max Pooling  
  - Fully Connected Layers  
  - Softmax Output  

---

## 🔬 Explainable AI (Grad-CAM)

Grad-CAM provides **visual explanations** by highlighting regions in the MRI scan that influenced the model’s decision.

> ⚠️ *Note:* Grad-CAM shows model attention, not exact tumor boundaries.

---

## ⚙️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Report Generation | ReportLab |

---

## 🗂️ Project Structure

```bash
BrainTumorSystem/
├── model/
│   └── model_loader.py
├── utils/
│   ├── preprocess.py
│   ├── gradcam.py
│   └── report.py
├── assets/
│   └── ai_stamp.png
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔄 System Workflow

1. Upload MRI Image  
2. Image Quality Assessment  
3. Preprocessing (resize + normalization)  
4. Model Prediction  
5. Probability Distribution  
6. Confidence Analysis  
7. Grad-CAM Visualization  
8. Clinical Interpretation  
9. Report Generation  

---

## 📊 Outputs

- Tumor classification result  
- Confidence score  
- Probability distribution chart  
- Grad-CAM heatmap  
- Clinical interpretation  
- Downloadable PDF report  

---

## 📥 Model Download

Due to GitHub limitations, the trained model is not included.

👉 Download from:  
https://drive.google.com/drive/folders/1J6zwcEmjOlWpcxnOJCGMR1g0edaYCM2G?usp=sharing

Place it inside:

model/

---

## ▶️ Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/TanmayT134/Explainable-Brain-Tumor-Detection.git
cd Explainable-Brain-Tumor-Detection
```

### 2. Create Virtual Environment

python -m venv venv

### 3. Activate Environment

#### Windows

venv\Scripts\activate

#### Mac/Linux

source venv/bin/activate

### 4. Install Dependencies

pip install -r requirements.txt

### 5. Run App

streamlit run app.py

---

## 📈 Results

Accurate classification across tumor classes

Grad-CAM highlights meaningful regions

Provides explainable predictions

Generates structured diagnostic reports

---

## 📸 Application Preview

### 🖥️ User Interface
![UI](assets/ui_main.png)

---

### 📊 Prediction Output
![Prediction](assets/prediction_output.png)

---

### 🔥 Grad-CAM Visualization
![GradCAM](assets/gradcam_output.png)

---

### 📄 Generated Report
![Report](assets/report_output.png)

---

### 🧪 Preprocessing Steps
![Preprocessing](assets/preprocessing.png)

---

## ⚠️ Limitations

Trained on limited dataset

Not intended for clinical use

Grad-CAM provides approximate explanations

---

## 🎯 Applications

AI-assisted medical imaging

Educational tool for medical AI

Explainable AI research

Computer-aided diagnosis

--- 

## 👨‍💻 Team & Contributions

This project was developed collaboratively with clearly defined responsibilities:

### 🧠 Tanmay Tawade *(Lead Developer)*
- Designed and implemented the complete system architecture  
- Developed CNN-based prediction pipeline  
- Integrated Grad-CAM for explainable AI  
- Built Streamlit web application (UI/UX)

---

### 📊 Aishwarya Kale *(Data & Documentation)*
- Dataset collection and preprocessing  
- Data organization and validation  
- Report writing and documentation  

---

### 🧪 Sakshi Bedekar *(Research & Testing)*
- Project ideation and conceptual design  
- Model testing and result evaluation  
- Presentation materials and validation  

> 📌 *All major design decisions and improvements were discussed and finalized collaboratively.*

---

## ⭐ Acknowledgements

Kaggle Brain MRI Dataset

TensorFlow & Keras

Streamlit

Research papers on CNN & XAI
