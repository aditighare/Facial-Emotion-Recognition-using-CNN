import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

# Load the pre-trained model
classifier = load_model(r'model.h5')  # Replace 'model.h5' with your model file

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the image
image_path = (r"C:\Users\HP\Pictures\Camera Roll\WIN_20240509_00_15_56_Pro.jpg")
# Replace with your image path
img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')

# Preprocess the image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize pixel values

# Make the prediction
prediction = classifier.predict(img_array)[0]
predicted_emotion = emotion_labels[prediction.argmax()]

# Display the result
print(f"Predicted Emotion: {predicted_emotion}")


'''
# ... (previous code)

# Optionally, display the image with predicted emotion
img_display = (img_array[0] * 255).astype(np.uint8)  # Convert to uint8 for display
cv2.putText(img_display, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('Image with Prediction', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

# Display the result with improvements
img_display = (img_array[0] * 255).astype(np.uint8)

# Convert to color if needed (assuming original image was color)
if img_display.shape[-1] == 1:  # Check if grayscale
    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

# Resize for better visualization
#img_display = cv2.resize(img_display, (300, 300))  # Adjust size as needed
img_display = cv2.resize(img_display, (500, 500), interpolation=cv2.INTER_AREA)  # For downscaling

# Put text with adjusted position
cv2.putText(img_display, predicted_emotion, (10, img_display.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Image with Prediction', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()