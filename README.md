# Handwritten Digit Generator (MNIST)

This app allows users to select a digit (0–9) and generates 5 unique images of that digit using a Conditional GAN trained on the MNIST dataset.

Features
- Selectable digit from dropdown
- Generates and displays 5 handwritten-style images
- Built with Streamlit + PyTorch

How to Run
1. Clone this repo
2. Upload your trained `generator.pth` file
3. Run with Streamlit:

bash
streamlit run app.py
