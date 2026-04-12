from tensorflow.keras.models import load_model, clone_model
import os

def load_cnn_model():
    model_path = os.path.join("model", "multiclass_brain_tumor_cnn.h5")

    original_model = load_model(model_path, compile=False)

    # 🔥 Clone model to rebuild graph
    model = clone_model(original_model)
    model.set_weights(original_model.get_weights())

    print("✅ Model cloned and graph rebuilt")

    return model