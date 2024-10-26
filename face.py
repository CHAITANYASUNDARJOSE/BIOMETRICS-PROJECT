import numpy as np
from keras.models import load_model

def load_face_recognition_model(model_path='face_recognition_model.h5'):
    model = load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    return image

def authenticate_face(model, image1_path, image2_path):
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    similarity = model.predict([image1, image2])
    return similarity[0][0]

# Example usage
model = load_face_recognition_model()
similarity_score = authenticate_face(model, 'thermal_face.jpg', 'thermal_face_to_authenticate.jpg')

if similarity_score > 0.8:  # Threshold for authentication
    print("Authenticated")
else:
    print("Not Authenticated")
