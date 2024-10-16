import cv2
import numpy as np
import os


def preprocess_image(image_path):
    """
    Preprocess
    """

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Cannot read image from path: {image_path}")
        return None, None

    blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=17,
        C=3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return image_gray, thresh


def preprocess_and_save_images(input_dir, output_dir):
    """
    Preprocess and save all images in the input directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            image_color, thresh = preprocess_image(input_path)

            if image_color is not None:
                cv2.imwrite(output_path, thresh)
                print(f"Saved preprocessed image: {output_path}")
            else:
                print(f"Failed to process image: {input_path}")

if __name__ == "__main__":

    input_image_folder = r"D:\pre pe\code\trainingcrew\dataset\images\train"
    output_image_folder = r"D:\pre pe\code\trainingcrew\dataset\images\preprocessed_train"

    preprocess_and_save_images(input_image_folder, output_image_folder)