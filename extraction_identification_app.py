import extraction_identification_model
import streamlit as st
import cv2
import numpy as np




def main():
    st.title("Image Extraction App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        extracted_text, language = extraction_identification_model.extract(image)

        st.header("Extracted Text:")
        st.text(extracted_text)
        st.header("Language OF Text:")
        st.text(language[0])


if __name__ == "__main__":
    main()
