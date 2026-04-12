import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import uuid
import tempfile
import os

from model.model_loader import load_cnn_model
from utils.preprocess import preprocess_image
from utils.report import generate_report
from utils.gradcam import get_gradcam_heatmap, overlay_heatmap


# ==========================
# MODEL LOAD (CACHED)
# ==========================
@st.cache_resource
def load_model_cached():
    return load_cnn_model()

model = load_model_cached()


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Brain Tumor AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================
# 🎨 GLOBAL CSS
# ==========================
st.markdown("""
<style>

/* BACKGROUND */
body {
    background-color: #0E1117;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    margin: auto;
}

/* TITLE */
.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #4CAF50, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.sub-title {
    font-size: 18px;
    color: #BBBBBB;
    margin-bottom: 25px;
}

/* 🔥 IMPROVED CARD */
.card {
    background: #161B22;   /* darker solid */
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #2A2F3A;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}

/* CARD HOVER (SUBTLE) */
.card:hover {
    border: 1px solid #4CAF50;
}

/* TEXT */
h2, h3 {
    color: #EAEAEA !important;
}

/* METRICS */
[data-testid="metric-container"] {
    background: #1E242D;
    border-radius: 10px;
    padding: 10px;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4CAF50, #00E5FF);
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #4CAF50, #00E5FF);
    color: white;
    border: none;
    font-weight: bold;
}

/* SIDEBAR FIX */
section[data-testid="stSidebar"] {
    background-color: #11161C;
    border-right: 1px solid #2A2F3A;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# 🏥 HEADER
# ==========================
st.markdown('<div class="main-title">🧠 Brain MRI AI Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning • Explainable AI • Clinical Visualization</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================
# 🏥 SIDEBAR
# ==========================
st.sidebar.markdown("## 🧠 Model Dashboard")

st.sidebar.markdown("""
### ⚙️ Model
- CNN Architecture  
- Input: 224 × 224  

### 🧬 Classes
- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

### 🔬 Features
- Grad-CAM Explainability  
- Confidence Analysis  

---

⚠️ *For Research and Educational use only. Not for clinical use*
""")

# ==========================
# 📤 FILE UPLOAD
# ==========================
uploaded_files = st.file_uploader(
    "Upload MRI Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ==========================
# MAIN LOOP
# ==========================
if uploaded_files:
    for uploaded_file in uploaded_files:

        st.markdown("---")
        st.subheader(f"📁 Processing: {uploaded_file.name}")

        try:
            image = Image.open(uploaded_file)
        except:
            st.error("Invalid image")
            continue

        img = np.array(image)

        if len(img.shape) not in [2, 3]:
            st.error("Unsupported format")
            continue

        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Temp image
        uid = str(uuid.uuid4())
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image.name)
        image_path = temp_image.name


        # ==========================
        # 🔍 IMAGE QUALITY
        # ==========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Image Quality Check")

        if img.mean() < 50:
            st.warning("Image appears too dark")
        elif img.mean() > 200:
            st.warning("Image appears too bright")
        else:
            st.success("Image quality is acceptable")

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 🧪 PREPROCESSING
        # ==========================
        processed, steps = preprocess_image(img)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Preprocessing Steps")

        cols = st.columns(len(steps))
        for i, (title, step_img) in enumerate(steps.items()):
            with cols[i]:
                if step_img.max() <= 1:
                    st.image((step_img * 255).astype(np.uint8), caption=title)
                else:
                    st.image(step_img, caption=title)

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 🧠 PREDICTION
        # ==========================
        prediction = model(processed).numpy()

        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = class_labels[predicted_index]

        result = "No Tumor Detected" if predicted_label == "notumor" else "Tumor Detected"


        # ==========================
        # 📊 PROBABILITIES
        # ==========================
        probabilities = prediction[0] * 100

        df = pd.DataFrame({
            "Tumor": class_labels,
            "Probability (%)": probabilities
        })

        prob_dict = {
            class_labels[i]: float(probabilities[i])
            for i in range(len(class_labels))
        }

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Class Probabilities")
        st.bar_chart(df.set_index("Tumor"))
        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # ⚠️ CONFUSION CHECK
        # ==========================
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_probs) > 1 and (sorted_probs[0][1] - sorted_probs[1][1]) < 15:
            st.warning(f"Model may be confused between {sorted_probs[0][0]} and {sorted_probs[1][0]}")


        # ==========================
        # 🧠 RESULT CARD
        # ==========================
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🧠 Diagnosis Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Result", result)
        col2.metric("Tumor Type", predicted_label)
        col3.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(float(confidence))

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 🧠 CONFIDENCE
        # ==========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Confidence Analysis")

        if confidence > 0.9:
            st.success("High Confidence Prediction")
        elif confidence > 0.7:
            st.warning("Moderate Confidence")
        else:
            st.error("Low Confidence")

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 🔥 GRAD-CAM
        # ==========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Model Explanation")

        heatmap = get_gradcam_heatmap(model, processed, "conv2d_2")
        heatmap_path = None

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original MRI", use_container_width=True)

        if heatmap is not None:
            gradcam_image = overlay_heatmap(heatmap, img)

            with col2:
                st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)

            st.caption("Highlighted regions indicate areas influencing prediction, not exact tumor boundaries.")

            temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_heatmap.name, cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR))
            heatmap_path = temp_heatmap.name
        else:
            with col2:
                st.warning("Grad-CAM not available")

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 📋 INTERPRETATION
        # ==========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Interpretation")

        if result == "No Tumor Detected":
            st.success("MRI appears normal.")
        else:
            st.error(f"{predicted_label} tumor detected. Clinical verification recommended.")

        st.markdown('</div>', unsafe_allow_html=True)


        # ==========================
        # 📄 REPORT
        # ==========================
        if st.button(f"Generate Report ({uploaded_file.name})"):

            report_path = generate_report(
                image_path,
                heatmap_path,
                result,
                predicted_label,
                confidence * 100,
                prob_dict
            )

            with open(report_path, "rb") as file:
                st.download_button(
                    "Download Report",
                    file,
                    file_name=f"report_{uploaded_file.name}.pdf"
                )

            # Cleanup
            try:
                os.remove(image_path)
                if heatmap_path:
                    os.remove(heatmap_path)
                os.remove(report_path)
            except:
                pass
            
        
        st.markdown("---")
        st.markdown(
            "<center style='color: gray;'>AI-Based Brain Tumor Detection System • Built with Streamlit</center>",
            unsafe_allow_html=True
        )