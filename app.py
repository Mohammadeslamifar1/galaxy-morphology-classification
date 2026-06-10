import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.predict import load_prediction_model, predict_image
from src.gradcam import create_gradcam_overlay


st.set_page_config(
    page_title="Galaxy Morphology Classification",
    page_icon="🌌",
    layout="wide",
)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_prediction_model(device=device)
    return model, device


st.title("Galaxy Morphology Classification")

st.markdown(
    """
    This app classifies galaxy morphology from an uploaded image using a ResNet18 deep learning model.

    The model was fine tuned on the Galaxy10 SDSS dataset, which contains galaxy images from the Sloan Digital Sky Survey with labels from Galaxy Zoo.
    """
)

st.sidebar.title("Project Information")

st.sidebar.markdown(
    """
    **Model:** ResNet18 transfer learning  
    **Dataset:** Galaxy10 SDSS  
    **Input:** Galaxy image  
    **Output:** Morphology class prediction  
    """
)

st.sidebar.markdown("### Notes")
st.sidebar.markdown(
    """
    The model works best with galaxy images similar to the Galaxy10 SDSS dataset.
    
    Rare classes may be more difficult because the dataset is imbalanced.
    """
)

model, device = load_model()

st.sidebar.success(f"Using device: {device}")

uploaded_file = st.file_uploader(
    "Upload a galaxy image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    predictions = predict_image(
        image=image,
        model=model,
        device=device,
        top_k=3,
    )

    top_prediction = predictions[0]
    
    gradcam_overlay = create_gradcam_overlay(
    image=image,
    model=model,
    device=device,
    target_class_index=top_prediction["class_id"],
)

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Galaxy Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Class",
            value=top_prediction["class_name"],
        )

        st.metric(
            label="Confidence",
            value=f"{top_prediction['confidence'] * 100:.2f}%",
        )

        st.subheader("Top 3 Predictions")

        prediction_table = pd.DataFrame(
            [
                {
                    "Class": prediction["class_name"],
                    "Confidence": f"{prediction['confidence'] * 100:.2f}%",
                }
                for prediction in predictions
            ]
        )

        st.table(prediction_table)

        st.bar_chart(
            pd.DataFrame(
                {
                    "Confidence": [
                        prediction["confidence"] for prediction in predictions
                    ]
                },
                index=[prediction["class_name"] for prediction in predictions],
            )
        )
        st.subheader("Grad CAM Explanation")
        st.image(
            gradcam_overlay,
            caption="Highlighted regions show areas that influenced the model prediction.",
            use_container_width=True,
        )

else:
    st.info("Upload a galaxy image to start prediction.")