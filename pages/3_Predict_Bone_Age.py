import streamlit as st
from PIL import Image
import torch
from model import load_model, preprocess_image, is_probable_xray
import pandas as pd

st.title("Predict Bone Age with Confidence Score")


@st.cache_resource
def get_model_and_info():
    checkpoint = torch.load("best_resnet50_model.pth", map_location="cpu")
    model = load_model("best_resnet50_model.pth")
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    return model, epoch, val_loss


model, epoch, val_loss = get_model_and_info()

st.info(f"Model trained until **epoch {epoch}** with final validation loss **{val_loss}**")


uploaded_files = st.file_uploader(
    "Upload Hand X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')

        if not is_probable_xray(image):
            st.error(f"File {uploaded_file.name} does not seem like a hand X-ray. Please upload a valid X-ray image.")
            continue

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            pred = model(input_tensor).item()
        results.append({"filename": uploaded_file.name,
                        "predicted_age_months": round(pred, 2),
                        "image": image
        })

    # Show results only if there is at least one valid prediction
    if results:
        for r in results:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(r["image"], width=100)
            with cols[1]:
                st.write(f"**{r['filename']}**")
                st.write(f"Predicted Age: **{r['predicted_age_months']} months**")

        # Optional: Download CSV without images
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image'} for r in results])
        csv = df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", data=csv, file_name="boneage_predictions.csv",
                               mime="text/csv")
    else:
        st.info("No valid hand X-ray images were uploaded.")
