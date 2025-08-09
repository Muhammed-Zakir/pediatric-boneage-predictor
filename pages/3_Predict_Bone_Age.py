import streamlit as st
from PIL import Image
import torch
from model import load_model, preprocess_image
import pandas as pd
from PIL import ImageStat
import time

st.title("Predict Bone Age with Confidence Score")


@st.cache_resource
def get_model_and_info():
    checkpoint = torch.load("best_resnet50_model.pth", map_location="cpu")
    model = load_model("best_resnet50_model.pth")
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    return model, epoch, val_loss


def is_valid_image_size(image: Image.Image, min_size=200, max_size=3000, aspect_ratio_tol=0.3):
    """
    Check if image size is within [min_size, max_size] for width and height
    and if aspect ratio (width/height) is within tolerance of expected ratio (approx 1:1 here).

    Returns True if valid.
    """
    w, h = image.size
    if w < min_size or h < min_size or w > max_size or h > max_size:
        return False
    aspect_ratio = w / h
    if abs(aspect_ratio - 1) > aspect_ratio_tol:  # expecting roughly square images
        return False
    return True


def is_probable_xray(image: Image.Image, threshold=0.3):
    """
    Simple heuristic: calculate ratio of pixels with brightness in
    typical X-ray range (dark + midtones) vs total pixels.
    Returns True if ratio above threshold.
    """
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0] / 255.0  # Normalize 0-1

    # If mean brightness is too high or too low, probably not X-ray
    if mean_brightness < 0.1 or mean_brightness > 0.6:
        return False
    return True


model, epoch, val_loss = get_model_and_info()

st.info(f"Model trained until **epoch {epoch}** with final validation loss **{val_loss}**")


uploaded_files = st.file_uploader(
    "Upload Hand X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

progress_bar = st.progress(0)
status_text = st.empty()

total = len(uploaded_files)

if uploaded_files:
    results = []
    error_messages = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')

        #  Image size validation
        if not is_valid_image_size(image):
            error_messages.append(uploaded_file.name)
            continue
        #  Image
        if not is_probable_xray(image):
            error_messages.append(uploaded_file.name)
            continue

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            pred = model(input_tensor).item()
            time.sleep(0.5)  # simulate prediction
            results.append(f"Predicted: {image}")
        results.append({
            "filename": uploaded_file.name,
            "predicted_age_months": round(pred, 2),
            "image": image
        })

        # Update progress bar
        progress = int(i / total * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i} of {total} images...")

    status_text.text("✅ Batch prediction completed!")

    # Show predictions first
    if results:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.write(f"**{len(results)}** / **{len(uploaded_files)}** predicted")
        st.write("")
        for r in results:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(r["image"], width=200)
            with cols[1]:
                st.write(f"**{r['filename']}**")
                st.write(f"Predicted Age: **{r['predicted_age_months']} months**")

        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image'} for r in results])
        csv = df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", data=csv, file_name="boneage_predictions.csv", mime="text/csv")
    else:
        st.info("No valid hand X-ray images were uploaded.")

    # Then show all errors after predictions
    if error_messages:
        st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
        st.markdown("The following images did not look like valid hand X-rays:")
        for filename in error_messages:
            st.write(f"- {filename}")

