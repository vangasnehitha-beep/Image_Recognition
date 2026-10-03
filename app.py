import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")


# ==========================================================
# INTERNAL CSS
# ==========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7ff, #eef8ff);
    }

    /* Main title */
    h1 {
        color: #243b6b;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Normal text */
    p {
        color: #4b5563;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf2ff, #f7fbff);
        border-right: 1px solid #dbe5f1;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #243b6b;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #405070;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.75);
        border: 2px dashed #93c5fd;
        border-radius: 15px;
        padding: 10px;
    }

    /* Image */
    [data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        font-weight: 700;
        padding: 10px 20px;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
    }

    /* Success message */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Prediction heading */
    h2, h3 {
        color: #243b6b;
    }

    /* Progress bars */
    [data-testid="stProgressBar"] > div > div {
        border-radius: 10px;
    }

    /* Camera */
    [data-testid="stCameraInput"] {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Info box */
    [data-testid="stAlertContainer"] {
        border-radius: 14px;
    }

</style>
""", unsafe_allow_html=True)


MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50",
}


@st.cache_resource(show_spinner="Loading model (first run downloads it, please wait)...")
def load_classifier(model_id: str):
    return pipeline("image-classification", model=model_id)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Models are trained on ImageNet (1,000 everyday object classes).")

# ---------------- Main ----------------
st.title("🖼️ Image Recognition")
st.write("Upload an image or take a photo, and the model will tell you what it sees.")

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
image = None

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file:
        image = Image.open(file).convert("RGB")

with tab_camera:
    photo = st.camera_input("Take a picture")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    st.image(image, caption="Input image")

    if st.button("🔍 Recognize", type="primary"):
        classifier = load_classifier(MODELS[model_name])
        with st.spinner("Analyzing image..."):
            results = classifier(image, top_k=top_k)

        best = results[0]
        st.success(f"**Top prediction:** {best['label']}  ({best['score'] * 100:.2f}%)")

        st.subheader("All predictions")
        for r in results:
            st.write(f"{r['label']} — {r['score'] * 100:.2f}%")
            st.progress(float(r["score"]))
else:
    st.info("👆 Upload an image or use the camera to get started.")
