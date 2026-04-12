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
            heatmap /= np.max(heatmap)

        return heatmap

    except Exception as e:
        print("Grad-CAM Error:", e)
        return None


# =========================================
# OVERLAY
# =========================================
def overlay_heatmap(heatmap, image):

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 🔥 CLEAN NOISE (CORRECT PLACE)
    heatmap = np.where(heatmap > 0.6, heatmap, 0)

    # Resize
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Scale to 0–255
    heatmap = np.uint8(255 * heatmap)

    # Color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend
    return cv2.addWeighted(image.astype(np.uint8), 0.6, heatmap, 0.4, 0)


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