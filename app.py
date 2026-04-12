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

# Load model
@st.cache_resource
def load_model_cached():
    from model.model_loader import load_cnn_model
    return load_cnn_model()

model = load_model_cached()

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# ==========================
# 🏥 HEADER
# ==========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='margin-bottom: 0;'>🧠 AI-Based Brain Tumor Detection System</h1>
<p style='font-size:18px; color:gray; margin-top:5px;'>
Deep Learning Powered MRI Analysis with Explainable AI
</p>
""", unsafe_allow_html=True)

st.info("📌 Upload an MRI scan to detect tumor presence and visualize model attention using Grad-CAM.")

st.sidebar.markdown("## 🧠 Model Information")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Model Details")
st.sidebar.markdown("""
- **Type:** CNN (Sequential)
- **Input:** 224 × 224 × 3
- **Output Classes:** 4
""")

st.sidebar.markdown("### 🧬 Tumor Classes")
st.sidebar.markdown("""
- Glioma  
- Meningioma  
- No Tumor  
- Pituitary  
""")

st.sidebar.markdown("### ⚙️ System Info")
st.sidebar.markdown("""
- Task: Multiclass Classification  
- Dataset: Brain MRI Images  
- Explainability: Grad-CAM  
""")

st.sidebar.markdown("---")

st.sidebar.warning("For educational use only. Not a clinical diagnosis tool.")

# Upload
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
        st.subheader(f"Processing: {uploaded_file.name}")

        # Load
        try:
            image = Image.open(uploaded_file)
        except:
            st.error("Invalid image")
            continue

        img = np.array(image)

        # Validate
        if len(img.shape) not in [2, 3]:
            st.error("Unsupported format")
            continue

        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save unique image
        uid = str(uuid.uuid4())
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image.name)

        image_path = temp_image.name

        # ==========================
        # IMAGE QUALITY CHECK
        # ==========================
        st.subheader("🔍 Image Quality Check")

        if img.mean() < 50:
            st.warning("Image appears too dark")
        elif img.mean() > 200:
            st.warning("Image appears too bright")
        else:
            st.success("Image quality is acceptable")

        # ==========================
        # PREPROCESS
        # ==========================
        processed, steps = preprocess_image(img)

        st.subheader("🧪 Preprocessing Steps")
        cols = st.columns(len(steps))

        for i, (title, step_img) in enumerate(steps.items()):
            with cols[i]:
                if step_img.max() <= 1:
                    st.image((step_img * 255).astype(np.uint8), caption=title)
                else:
                    st.image(step_img, caption=title)

        # ==========================
        # PREDICTION
        # ==========================
        prediction = model(processed).numpy()

        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = class_labels[predicted_index]

        result = "No Tumor Detected" if predicted_label == "notumor" else "Tumor Detected"

        # ==========================
        # PROBABILITY GRAPH
        # ==========================
        st.subheader("📊 Class Probabilities")

        probabilities = prediction[0] * 100

        df = pd.DataFrame({
            "Tumor": class_labels,
            "Probability (%)": probabilities
        })

        st.bar_chart(df.set_index("Tumor"))

        prob_dict = {
            class_labels[i]: float(probabilities[i])
            for i in range(len(class_labels))
        }

        # ==========================
        # CONFUSION CHECK
        # ==========================
        st.subheader("⚠️ Possible Misclassification")

        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_probs) > 1 and (sorted_probs[0][1] - sorted_probs[1][1]) < 15:
            st.warning(f"Model may be confused between {sorted_probs[0][0]} and {sorted_probs[1][0]}")

        # ==========================
        # RESULTS
        # ==========================
        st.subheader("🧠 Diagnosis Result")
        st.write("Result:", result)
        st.write("Tumor Type:", predicted_label)
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.progress(float(confidence))

        # ==========================
        # CONFIDENCE SYSTEM
        # ==========================
        st.subheader("🧠 Confidence Analysis")

        if confidence > 0.9:
            st.success("High Confidence Prediction")
        elif confidence > 0.7:
            st.warning("Moderate Confidence")
        else:
            st.error("Low Confidence")

        # Uncertainty
        if confidence < 0.7:
            st.warning("""
Possible reasons:\n
• Poor image quality  
• Unseen patterns  
• Model limitation  
""")

        # ==========================
        # 🔥 MODEL EXPLANATION (SIDE BY SIDE)
        # ==========================
        st.subheader("🔥 Model Explanation")

        original_img = img.copy()
        heatmap_path = None

        heatmap = get_gradcam_heatmap(model, processed, "conv2d_2")

        if heatmap is not None:
            gradcam_image = overlay_heatmap(heatmap, original_img)

            col1, col2 = st.columns(2)

            with col1:
                st.image(original_img, caption="Original MRI", width="stretch")

            with col2:
                st.image(gradcam_image, caption="Model Attention Map (Grad-CAM)", width="stretch")

            # Interpretation text
            if predicted_label == "notumor":
                st.info("Model attention shows regions analyzed in a normal brain scan.")
            else:
                st.info("Highlighted regions influenced tumor detection.")

            # Save for report
            temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

            cv2.imwrite(
                temp_heatmap.name,
                cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR)
            )

            heatmap_path = temp_heatmap.name

        else:
            st.warning("Grad-CAM failed")

        # ==========================
        # INTERPRETATION
        # ==========================
        st.subheader("📋 Interpretation")

        if result == "No Tumor Detected":
            st.success("MRI appears normal.")
        else:
            st.error(f"{predicted_label} tumor detected. Clinical verification recommended.")

        # ==========================
        # LIMITATIONS
        # ==========================
        st.subheader("⚠️ Model Limitations")

        st.warning("""
• Trained on limited dataset  
• Not for clinical use  
• Grad-CAM is approximate  
""")

        # ==========================
        # REPORT
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
            
            # Cleanup temp files
            try:
                os.remove(image_path)
                if heatmap_path:
                    os.remove(heatmap_path)
                os.remove(report_path)
            except:
                pass