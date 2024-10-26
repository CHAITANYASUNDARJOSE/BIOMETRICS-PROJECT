import cv2
import numpy as np

def mse(imageA, imageB):
    # Compute Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(image1, image2):
    # Resize images to the same size if necessary
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Compute MSE between the two images
    error = mse(image1, image2)
    return error

def capture_live_image(camera_index=0, output_path='live_image.jpg'):
    # Capture live image from webcam
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if ret:
        # Convert frame to grayscale and then apply thermal-like effect
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, thermal_frame)
        cap.release()
        return thermal_frame
    else:
        cap.release()
        raise Exception("Failed to capture live image.")

def authenticate_user(stored_image_path='C:\\phone\\engineering\\sem 7\\biometrics\\project\\image2.jpg', live_image_path='live_image.jpg'):
    # Load stored image
    stored_image = cv2.imread(stored_image_path)

    # Capture live thermal image
    live_image = capture_live_image(output_path=live_image_path)

    # Compare the stored image and live image
    error = compare_images(stored_image, live_image)
    
    # Set a threshold for authentication
    if error < 35000:  # Adjust based on your testing
        print("Authenticated successfully!")
    else:
        print("Authentication failed. Images do not match.")

# Example usage
authenticate_user()
