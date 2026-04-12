import cv2
import numpy as np

def preprocess_image(image):

    steps = {}

    steps["Original"] = image.copy()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    steps["RGB"] = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = image[y:y+h, x:x+w]
        
        steps["Cropped Brain"] = cropped.copy()

        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_crop)

        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        steps["Contrast Enhanced"] = enhanced.copy()
        
        smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)

        steps["Smoothed"] = smoothed.copy()
    else:
        cropped = image

    resized = cv2.resize(smoothed, (224, 224))
    steps["Resized"] = resized.copy()
    
    normalized = resized.astype("float32") / 255.0
    steps["Normalized"] = normalized.copy()

    final = np.expand_dims(normalized, axis=0)

    return final, steps