
# **📚 AI-School-Helper: Enhanced Image-to-Text Extraction for All Subjects**

Welcome to AI-School-Helper, a project that includes two Python scripts—**school2.py** and **school.py**—designed to make extracting text from images easier and more accurate across all subjects, with special enhancements for complex topics like math and physics.

## **🚀 Features**

• **🖼️ Image Source Selection:** Choose between selecting an existing image file or capturing one from your camera.

• **📷 Resizable Camera Window:** Capture images using a resizable camera window, optimized for your screen size.

• **🧠 Dual OCR Models:**

- **LaTeXOCR (only on school2.py):** Perfect for extracting complex formulas and equations from Math, Physics, and Chemistry images.

- **EasyOCR:** Ideal for extracting text from general subjects like English, Coding, History, and other non-math subjects.

• **🤖 Groq Integration:** Ask questions about the extracted text using the powerful **LLaMA 3 language model**.
## **🛠️ Installation**

Make sure to install the required packages before running the scripts:

    pip install opencv-python-headless pillow pix2tex groq easyocr tkinter screeninfo



## **📄 How to Use**

1.  **Choose an image source:**


- Option 1: Select an existing image file.

- Option 2: Capture an image using your webcam.

2. **Select an OCR model:**

- **LaTeXOCR (only on school2.py):** For Math, Physics, and Chemistry. 

- **EasyOCR:** For English, Coding, History, and other subjects.

3. **Interact with the AI:**

Ask questions about the extracted text, and get responses using **Groq's LLaMA 3 model**.
## **🆕 Improvements in school2.py**

**Advanced OCR:** The **school2.py** script leverages **LaTeXOCR** for better accuracy with mathematical and scientific content, making it an improved version of **school.py**.

**Enhanced User Experience:** A more intuitive interface and better error handling.

## **👏 Credits**

Special thanks to the creators of [LaTeXOCR]([url](https://github.com/lukas-blecher/LaTeX-OCR)) for making accurate text extraction from complex scientific images possible.

## **Contribution**

Contribution of any type is well appericated!!!
