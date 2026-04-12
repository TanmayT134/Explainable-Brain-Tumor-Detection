import numpy as np
import tensorflow as tf
import cv2


def get_gradcam_heatmap(model, image, last_conv_layer_name):
    try:
        image = tf.convert_to_tensor(image)

        with tf.GradientTape() as tape:
            x = image
            conv_output = None

            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv_layer_name:
                    conv_output = x

            # 🔥 Ensure gradient tracking
            tape.watch(conv_output)

            predictions = x
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)

        if grads is None:
            print("Gradients are None")
            return None

        # Global Average Pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]

        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

        # ReLU
        heatmap = tf.maximum(heatmap, 0)

        heatmap = heatmap.numpy()

        # Normalize
        if np.max(heatmap) != 0:
            heatmap = heatmap - np.min(heatmap)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        print("Grad-CAM Error:", e)
        return None


# =========================================
# OVERLAY
# =========================================
def overlay_heatmap(heatmap, image):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # 🔥 Boost strong regions
    heatmap = np.power(heatmap, 2.5)

    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

    # 🔥 Keep only strongest activations
    threshold = np.percentile(heatmap, 92)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    # Brain mask
    brain_mask = get_brain_mask(image) / 255.0
    heatmap = heatmap * brain_mask

    # Convert to color
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay only active regions
    overlay = image.copy()
    mask = heatmap > 0

    overlay[mask] = cv2.addWeighted(image[mask], 0.5, colored[mask], 0.5, 0)

    return overlay

# =========================================
# OPTIONAL (NOT USED NOW)
# =========================================
def get_bounding_box(heatmap, threshold=0.6):

    heatmap_uint8 = np.uint8(255 * heatmap)

    _, thresh = cv2.threshold(
        heatmap_uint8,
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    return cv2.boundingRect(cnt)

def get_brain_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize contrast
    gray = cv2.equalizeHist(gray)

    # Strong threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Remove noise
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh