import streamlit as st
from PIL import Image, ImageStat
import torch
from model import load_model, preprocess_image
import pandas as pd
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Predict Bone Age with Confidence Score")

@st.cache_resource
def get_model_and_info():
    checkpoint = torch.load("best_resnet50_model.pth", map_location="cpu")
    model = load_model("best_resnet50_model.pth")
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    return model, epoch, val_loss

def is_valid_image_size(image: Image.Image, min_size=200, max_size=3000, aspect_ratio_tol=0.3):
    w, h = image.size
    if w < min_size or h < min_size or w > max_size or h > max_size:
        return False
    aspect_ratio = w / h
    if abs(aspect_ratio - 1) > aspect_ratio_tol:
        return False
    return True

def is_probable_xray(image: Image.Image, threshold=0.3):
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0] / 255.0
    if mean_brightness < 0.1 or mean_brightness > 0.6:
        return False
    return True

loaded_model, _epoch, _val_loss = get_model_and_info()

st.info(f"Model trained until **epoch {_epoch}** with final validation loss **{_val_loss}**")

uploaded_files = st.file_uploader(
    "Upload Hand X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

progress_bar = st.progress(0)
status_text = st.empty()

if uploaded_files:
    results = []
    error_messages = []

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert('RGB')

        # Validate size
        if not is_valid_image_size(image):
            error_messages.append(f"{uploaded_file.name} - Invalid size")
            continue

        # Validate if X-ray
        if not is_probable_xray(image):
            error_messages.append(f"{uploaded_file.name} - Not an X-ray")
            continue

        # Preprocess and predict
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            pred = loaded_model(input_tensor).cpu().item()

        # Save image as bytes
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Save results
        results.append({
            "filename": uploaded_file.name,
            "predicted_age_months": round(pred, 2),
            "image_bytes": image_bytes.getvalue()
        })

        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processed {i + 1} of {len(uploaded_files)} images")

    # Show predictions
    if results:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.write(f"**{len(results)}** / **{len(uploaded_files)}** predicted")
        for r in results:
            cols = st.columns([1, 2])
            with cols[0]:
                image = Image.open(BytesIO(r["image_bytes"]))
                st.image(image, width=200)
            with cols[1]:
                st.write(f"**{r['filename']}**")
                st.write(f"Predicted Age: **{r['predicted_age_months']} months**")

        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image_bytes'} for r in results])
        csv = df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", data=csv, file_name="boneage_predictions.csv", mime="text/csv")
    else:
        st.info("No valid hand X-ray images were uploaded.")

    # Show errors
    if error_messages:
        st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
        st.markdown("The following images did not look like valid hand X-rays:")
        for filename in error_messages:
            st.write(f"- {filename}")
