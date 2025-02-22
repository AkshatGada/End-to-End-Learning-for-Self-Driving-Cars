import tensorflow as tf
import imageio.v2 as imageio
import cv2
import numpy as np
from scipy.stats import pearsonr

dmodel_path = "save/model_epoch_29.h5"  
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

ground_truth = np.loadtxt("driving_dataset/ground_truth.txt")  # Ensure ground truth is in degrees

predictions = []

num_images = len(ground_truth)
for i in range(num_images):
    img_path = f"driving_dataset/data/{i}.jpg"  # Ensure correct path to images
    try:
        full_image = imageio.imread(img_path)
    except FileNotFoundError:
        print(f"File {img_path} not found, stopping inference.")
        break

   
    image = full_image[-150:]  
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the steering angle (model outputs in radians)
    angle_rad = model(image, training=False).numpy()[0][0]
    angle_deg = angle_rad * 180.0 / np.pi  # Convert radians to degrees
    predictions.append(angle_deg)

# Convert predictions list to a NumPy array for metric calculations
predictions = np.array(predictions)
gt = ground_truth  # alias for clarity

# 1. Mean Squared Error (MSE)
mse = np.mean((predictions - gt) ** 2)
print("Mean Squared Error (MSE):", mse)

# 2. Steering Angle Correlation (Pearson r)
corr, _ = pearsonr(predictions, gt)
print("Steering Angle Correlation (Pearson r):", corr)

# 3. Lane-Keeping Performance:
lane_threshold = 5.0  # degrees threshold (adjust as needed)
within_lane = np.sum(np.abs(predictions - gt) <= lane_threshold)
lane_keeping_performance = within_lane / len(gt) * 100
print("Lane-Keeping Performance (% within ±5°):", lane_keeping_performance)

# 4. Intervention Rate:
intervention_threshold = 10.0  # degrees threshold for abrupt changes
diffs = np.abs(np.diff(predictions))
interventions = np.sum(diffs > intervention_threshold)
intervention_rate = interventions / (len(predictions) - 1) * 100
print("Intervention Rate (% of frames with abrupt changes):", intervention_rate)
