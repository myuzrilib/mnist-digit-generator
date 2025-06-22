import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from generator import Generator

# Load model
model = Generator()
model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
model.eval()

st.title("Handwritten Digit Generator (MNIST)")

digit = st.selectbox("Select digit (0–9)", list(range(10)))
if st.button("Generate Images"):
    noise = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        images = model(noise, labels)

    grid = make_grid(images, nrow=5, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    st.pyplot(fig)
