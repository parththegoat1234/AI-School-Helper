import cv2
from PIL import Image
from pix2tex.cli import LatexOCR
from groq import Groq
import easyocr
import tkinter as tk
from tkinter import filedialog
import os
import logging
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)

def get_image_source():
    """Prompt the user to choose the image source."""
    while True:
        print("Choose image source:")
        print("1. Select existing image file")
        print("2. Capture image from camera")
        choice = input("Enter your choice (1 or 2): ")
        if choice in ['1', '2']:
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")

def get_image_path():
    """Open a file dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    return file_path

def capture_image_from_camera():
    """Capture an image from the camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open camera.")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Could not read frame.")
            break
        
        cv2.imshow("Press 'c' to capture or 'q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            cap.release()
            return frame
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    return None

def extract_text_with_latexocr(img):
    """Extract text from an image using LatexOCR."""
    try:
        model = LatexOCR()
        extracted_text = model(Image.fromarray(img))
        return extracted_text
    except Exception as e:
        logging.error(f"Error processing the image with LatexOCR: {e}")
        return None

def extract_text_with_easyocr(image_path):
    """Extract text from an image using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path)
        extracted_text = " ".join([text for (bbox, text, prob) in result])
        return extracted_text
    except Exception as e:
        logging.error(f"Error processing the image with EasyOCR: {e}")
        return None

def get_groq_client():
    """Initialize the Groq client with the provided API key."""
    api_key = "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"
    return Groq(api_key=api_key)

def main():
    client = get_groq_client()
    if not client:
        return

    choice = get_image_source()

    if choice == '1':
        print("Please select an image file to extract text from.")
        image_path = get_image_path()
        if not image_path:
            print("No image file selected. Exiting.")
            return
        try:
            img = cv2.imread(image_path)
        except Exception as e:
            logging.error(f"Error opening the image file: {e}")
            return
    elif choice == '2':
        print("Capturing image from camera...")
        img = capture_image_from_camera()
        if img is None:
            print("No image captured. Exiting.")
            return
        # Save captured image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image_path = tmp.name
            cv2.imwrite(image_path, img)
            print(f"Image saved to temporary file: {image_path}")  # Debug print
    else:
        print("Invalid choice. Exiting.")
        return

    print("Choose OCR model:")
    print("1. LatexOCR (for math, physics, chemistry)")
    print("2. EasyOCR (for English, coding, history, other languages)")
    ocr_choice = input("Enter your choice (1 or 2): ")

    if ocr_choice == '1':
        extracted_text = extract_text_with_latexocr(img)
    elif ocr_choice == '2':
        print(f"Using EasyOCR on image: {image_path}")  # Debug print
        extracted_text = extract_text_with_easyocr(image_path)
    else:
        print("Invalid choice. Exiting.")
        return

    if not extracted_text:
        return

    print("Extracted text:")
    print(extracted_text)

    while True:
        user_question = input("\nPlease enter your question about the extracted text (or type 'exit chat' to finish): ")
        if user_question.lower() == 'exit chat':
            print("Exiting chat.")
            break

        full_question = f"The extracted text from the image is: '{extracted_text}'. {user_question}"

        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": full_question
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            print("AI's response:")
            for chunk in completion:
                print(chunk.choices[0].delta.content or "", end="", flush=True)
            print("\n")

        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
