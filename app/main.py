import streamlit as st

st.image("static/logo.png", width=100)
st.title("DAISY - Dermatological AI System for You")
st.write("Welcome to DAISY! Upload a skin image to detect conditions (melanoma, bcc, nv, bkl, akiec).")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully! (Prediction feature coming soon)")