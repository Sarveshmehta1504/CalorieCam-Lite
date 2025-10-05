import streamlit as st
import torch, os, io, json, sys
import numpy as np
from PIL import Image
from torchvision import transforms

# Add the parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import EmbeddingClassifier
from src.data import get_transforms, load_calorie_map
from src.utils import load_json, device_auto
from src.explainability import GradCAM

# Configure Streamlit page (must be first Streamlit command)
st.set_page_config(
    page_title="CalorieCam Lite", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's deploy button and menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
.stActionButton {display: none;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("CalorieCam Lite — Few‑Shot Food Calorie Estimator")

ART_DIR = "artifacts/base_model"
CAL_CSV = "data/calorie_map.csv"

@st.cache_resource
def load_model_and_maps(art_dir):
    device = device_auto()
    label_map = load_json(os.path.join(art_dir,"label_map.json"))
    if label_map is None:
        st.error("No trained model found. Train the base model first.")
        st.stop()
    
    try:
        model = EmbeddingClassifier(num_classes=len(label_map)).to(device)
        ckpt = os.path.join(art_dir,"best.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(art_dir,"last.pt")
        if not os.path.exists(ckpt):
            st.error(f"Model checkpoint not found at {ckpt}")
            st.stop()
        
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        inv_label = {int(k):v for k,v in label_map.items()}

        # Few-shot prototypes
        protos = load_json(os.path.join(art_dir,"prototypes.json"), default={"classes":{}, "meta":{}})
        cal_map = load_calorie_map(CAL_CSV)
        return model, inv_label, protos, cal_map, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def to_tensor(img, size=224):
    tform = get_transforms(train=False, img_size=size)
    return tform(img).unsqueeze(0)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a)+1e-8)
    b = b / (np.linalg.norm(b)+1e-8)
    return float((a*b).sum())

def predict_linear(model, x, inv_label, device):
    with torch.no_grad():
        logits, emb = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return inv_label[pred_idx], probs[pred_idx], emb.cpu().numpy()[0], probs, pred_idx

def predict_prototype(model, x, inv_label, protos, device):
    # combine base classes as prototypes too (using classifier weights)
    with torch.no_grad():
        _, emb = model(x.to(device))
        emb = emb.cpu().numpy()[0]

    proto_dict = {}
    # add few-shot
    for cls, obj in protos.get("classes", {}).items():
        proto_dict[cls] = np.array(obj["vector"], dtype=np.float32)

    # add base classes via linear head weights as pseudo-prototypes
    W = model.head.fc.weight.detach().cpu().numpy()  # [C,512]
    for idx, name in inv_label.items():
        proto_dict[name] = W[idx]

    # nearest by cosine
    best_cls, best_sim = None, -1.0
    for k, v in proto_dict.items():
        s = cosine_sim(emb, v)
        if s > best_sim:
            best_cls, best_sim = k, s
    return best_cls, (best_sim+1)/2, emb  # map cosine [-1,1] to [0,1] as confidence

def overlay_cam(model, img_pil, x_tensor, class_idx):
    gc = GradCAM(model)
    import cv2, numpy as np
    orig = np.array(img_pil.convert("RGB"))
    heat = gc.generate(x_tensor, class_idx, orig)
    gc.close()
    return Image.fromarray(heat)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Inference Mode", ["Prototype (few‑shot + base)","Linear Head (base only)"], index=0)
    img_size = st.slider("Image size", 128, 384, 224, 32)

model, inv_label, protos, cal_map, device = load_model_and_maps(ART_DIR)

colL, colR = st.columns([2,1])
with colL:
    up = st.file_uploader("Upload a food image (jpg/png)", type=["jpg","jpeg","png"])
    if up is not None:
        try:
            img = Image.open(up).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
    else:
        st.info("Upload an image to get started.")
        st.stop()

    x = to_tensor(img, size=img_size)

    if mode.startswith("Prototype"):
        pred_label, conf, emb = predict_prototype(model, x, inv_label, protos, device)
        # pick class_idx for Grad-CAM from base classes if exists
        class_idx = None
        for k,v in inv_label.items():
            if v == pred_label:
                class_idx = k
                break
        st.subheader(f"Prediction: **{pred_label}**  •  Confidence: **{conf:.2f}**")
    else:
        pred_label, conf, emb, probs, class_idx = predict_linear(model, x, inv_label, device)
        st.subheader(f"Prediction: **{pred_label}**  •  Confidence: **{conf:.2f}**")

    kcal = cal_map.get(pred_label, None)
    if kcal is None:
        st.write("Estimated Calories: — (add this dish to `data/calorie_map.csv`)")
    else:
        st.write(f"Estimated Calories: **~{int(kcal)} kcal** (typical portion)")

    if class_idx is not None:
        with st.spinner("Computing Grad‑CAM..."):
            cam_img = overlay_cam(model, img, x.to(device), class_idx)
        st.image(cam_img, caption="Grad‑CAM heatmap", use_container_width=True)

with colR:
    st.write("**Prototype Classes Available:**")
    st.json(protos.get("classes", {}))
    st.write("**Base Classes:**")
    st.write(list(inv_label.values()))
    st.caption("Tip: add few‑shot classes via `src/adapt_fewshot.py`")

st.caption("© 2025 CalorieCam Lite — Demo project for educational use.")
