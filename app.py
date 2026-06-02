import streamlit as st
import cv2
import numpy as np
from PIL import Image
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
import os

load_dotenv()

st.set_page_config(
    page_title="Pulmonary Edema Ensemble System",
    layout="wide"
)

# Paths
DENSENET_W = "model/densenet121_weights.pth"
EFFNET_W = "model/efficientnet_weights.pth"
RESNET_W = "model/resnet50_weights.pth"
CONVNEXT_W = "model/convnext_weights.pth"

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    densenet_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 2)
    densenet_model.op_threshs = None
    densenet_model.load_state_dict(
        torch.load(DENSENET_W, map_location=device),
        strict=False
    )
    densenet_model = densenet_model.to(device).eval()

    resnet_model = models.resnet50()
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
    resnet_model.load_state_dict(
        torch.load(RESNET_W, map_location=device)
    )
    resnet_model = resnet_model.to(device).eval()

    effnet_model = models.efficientnet_b0()
    effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 2)
    effnet_model.load_state_dict(
        torch.load(EFFNET_W, map_location=device)
    )
    effnet_model = effnet_model.to(device).eval()
    
    convnext_model = models.convnext_tiny()
    convnext_model.classifier[2] = nn.Linear(convnext_model.classifier[2].in_features, 2)
    convnext_model.load_state_dict(
        torch.load(CONVNEXT_W, map_location=device)
    )
    convnext_model = convnext_model.to(device).eval()

    return densenet_model, resnet_model, effnet_model, convnext_model, device

densenet_model, resnet_model, effnet_model, convnext_model, device = load_models()

# Corrected class mapping: 0 -> Normal, 1 -> Edema
class_names = ['Normal', 'Edema']

# Prediction helpers
def validate_chest_xray_openai(image_path: str) -> dict:
    default_response = {
        "is_chest_xray": None,
        "confidence": 0.0,
        "view": "unknown",
        "reason": "OpenAI validation failed or was skipped.",
        "raw_response": ""
    }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        default_response["reason"] = "OpenAI validation is disabled because OPENAI_API_KEY is missing."
        return default_response

    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        default_response["reason"] = f"Failed to read image: {str(e)}"
        return default_response

    client = OpenAI(api_key=api_key)

    prompt = """You are validating an uploaded image for a medical imaging AI research demo.

Task:
Determine whether the uploaded image is a real chest X-ray radiograph.

Rules:
- Return true only if the image is clearly a chest X-ray radiograph.
- Accept AP, PA, or lateral chest X-rays.
- Reject normal photos, logos, documents, screenshots, drawings, CT scans, MRI scans, ultrasound images, pathology images, non-chest X-rays, and cropped non-diagnostic images.
- Do not diagnose disease.
- Do not mention pulmonary edema.
- Do not provide medical advice.
- Only validate image type.

Return strict JSON only:
{
  "is_chest_xray": true/false,
  "confidence": number between 0 and 1,
  "view": "AP" or "PA" or "lateral" or "unknown",
  "reason": "short explanation"
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        result["raw_response"] = content
        return result

    except Exception as e:
        default_response["reason"] = "OpenAI image validation could not be completed."
        default_response["raw_response"] = str(e)
        return default_response

class ApplyCLAHE(object):
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)

rgb_clahe = ApplyCLAHE()

transform_224 = transforms.Compose([
    rgb_clahe,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) 
])

# ConvNeXt preprocessing
convnext_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) 
])

xrv_base_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
    torchvision.transforms.Lambda(lambda x: torch.from_numpy(x).float())
])

def get_fused_heatmap(models_list, target_layers, input_tensors, original_img, target_category):
    all_cams = []
    for i in range(len(models_list)):
        cam_obj = GradCAM(model=models_list[i], target_layers=[target_layers[i]])
        grayscale_cam = cam_obj(
            input_tensor=input_tensors[i],
            targets=[ClassifierOutputTarget(target_category)]
        )[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
        all_cams.append(grayscale_cam)

    fused_cam = np.mean(all_cams, axis=0)
    img_vis = original_img.resize((224, 224))
    img_float = np.float32(img_vis) / 255.0
    return show_cam_on_image(img_float, fused_cam, use_rgb=True)

def get_single_model_heatmap(model, target_layer, input_tensor, original_img, target_category):
    cam_obj = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam_obj(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_category)]
    )[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
    img_vis = original_img.resize((224, 224))
    img_float = np.float32(img_vis) / 255.0
    return show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

# UI
st.title("AI-Pulmonary Edema Detector")
st.write("Ensemble AI Architecture: Integrating Multi-Model Intelligence.")

uploaded_file = st.file_uploader("Upload Chest X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name
        st.session_state["continue_after_validation_warning"] = False

    temp_path = "temp/temp_img.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Validating image..."):
        validation = validate_chest_xray_openai(temp_path)
        
    proceed_to_classification = True

    if validation["is_chest_xray"] is True:
        st.success("Chest X-ray validation passed.")
        view = validation.get('view', 'unknown')
        if view and view != "unknown":
            st.write(f"Detected view: {view}")
    elif validation["is_chest_xray"] is False:
        st.warning("Warning: The uploaded image may not be a valid chest X-ray. Please make sure you uploaded a cropped chest X-ray image.")
        st.write(f"Reason: {validation.get('reason', '')}")
        
        if not st.session_state.get("continue_after_validation_warning", False):
            proceed_to_classification = False
            if st.button("Continue Anyway"):
                st.session_state["continue_after_validation_warning"] = True
                st.rerun()
    elif validation["is_chest_xray"] is None:
        st.warning(validation.get('reason', 'OpenAI image validation could not be completed. Continuing with local model analysis.'))

    if not proceed_to_classification:
        st.stop()
        
    img_pil = Image.open(temp_path).convert('RGB')

    if "selected_heatmap" not in st.session_state:
        st.session_state["selected_heatmap"] = "Ensemble CAM"

    img_sk = skimage.io.imread(temp_path)
    if len(img_sk.shape) == 3:
        img_sk = img_sk.mean(2).astype(np.uint8)
    else:
        img_sk = img_sk.astype(np.uint8)

    img_clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img_sk)

    img_xrv = xrv_base_transform(
        xrv.datasets.normalize(img_clahe, 255)[None, ...]
    ).unsqueeze(0).to(device)

    img_224 = transform_224(img_pil).unsqueeze(0).to(device)
    img_convnext = convnext_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out_dense = F.softmax(densenet_model(img_xrv), dim=1)
        densenet_normal_prob = out_dense[0, 0].item()
        densenet_edema_prob = out_dense[0, 1].item()
        densenet_predicted_class = "Edema" if densenet_edema_prob >= 0.5 else "Normal"
        
        out_res = F.softmax(resnet_model(img_224), dim=1)
        resnet_normal_prob = out_res[0, 0].item()
        resnet_edema_prob = out_res[0, 1].item()
        resnet_predicted_class = "Edema" if resnet_edema_prob >= 0.5 else "Normal"
        
        out_eff = F.softmax(effnet_model(img_224), dim=1)
        effnet_normal_prob = out_eff[0, 0].item()
        effnet_edema_prob = out_eff[0, 1].item()
        effnet_predicted_class = "Edema" if effnet_edema_prob >= 0.5 else "Normal"
        
        out_convnext = F.softmax(convnext_model(img_convnext), dim=1)
        convnext_normal_prob = out_convnext[0, 0].item()
        convnext_edema_prob = out_convnext[0, 1].item()
        convnext_predicted_class = "Edema" if convnext_edema_prob >= 0.5 else "Normal"

        ensemble_edema_prob = (densenet_edema_prob + resnet_edema_prob + effnet_edema_prob + convnext_edema_prob) / 4.0
        
        if ensemble_edema_prob >= 0.5:
            final_predicted_class = "EDEMA"
            heatmap_target = 1
        else:
            final_predicted_class = "NORMAL"
            heatmap_target = 0

        # Model Agreement based on Edema Probabilities
        probs = [densenet_edema_prob, resnet_edema_prob, effnet_edema_prob, convnext_edema_prob]
        spread = max(probs) - min(probs)
        if spread < 0.15:
            agreement = "High"
        elif spread < 0.30:
            agreement = "Medium"
        else:
            agreement = "Low"

    with st.sidebar:
        st.header("Clinical Report")

        if final_predicted_class == "EDEMA":
            st.error("FINAL DIAGNOSIS: **EDEMA**")
        else:
            st.success("FINAL DIAGNOSIS: **NORMAL**")
            
        st.markdown(f"**Ensemble Edema Probability:**<br><span style='font-size:24px; color:#4CAF50;'>{ensemble_edema_prob*100:.2f}%</span>" if final_predicted_class == "NORMAL" else f"**Ensemble Edema Probability:**<br><span style='font-size:24px; color:#F44336;'>{ensemble_edema_prob*100:.2f}%</span>", unsafe_allow_html=True)
            
        st.markdown(f"**Model Agreement:** {agreement}")
        if agreement == "Low":
            st.warning("The models show disagreement. Interpret this result with caution.")

        st.markdown("---")
        st.subheader("Neural Network Votes")
        st.caption("Click a model to view its heatmap.")

        if st.button("Show Ensemble CAM", key="btn_ensemble", use_container_width=True):
            st.session_state["selected_heatmap"] = "Ensemble CAM"

        st.markdown("")

        def display_model_vote(name, edem_p, key_name):
            if edem_p >= 0.5:
                pred = "Edema"
                disp_p = edem_p
                label = "Edema Probability"
            else:
                pred = "Normal"
                disp_p = 1.0 - edem_p
                label = "Normal Probability"
                
            st.markdown(f"**{name}**<br>Prediction: {pred}", unsafe_allow_html=True)
            try:
                st.progress(float(disp_p), text=f"{label}: {disp_p*100:.1f}%")
            except TypeError:
                st.write(f"{label}: {disp_p*100:.1f}%")
                st.progress(float(disp_p))
            if st.button(f"View {name} CAM", key=f"btn_{key_name}", use_container_width=True):
                st.session_state["selected_heatmap"] = key_name
            st.markdown("<br>", unsafe_allow_html=True)
            
        display_model_vote("DenseNet121 XRV", densenet_edema_prob, "DenseNet121 XRV")
        display_model_vote("EfficientNet-B0", effnet_edema_prob, "EfficientNet-B0")
        display_model_vote("ResNet50", resnet_edema_prob, "ResNet50")
        display_model_vote("ConvNeXt-Tiny", convnext_edema_prob, "ConvNeXt-Tiny")

        st.markdown("---")
        st.warning("**Disclaimer:** This tool is for research purposes only and does not provide a medical diagnosis.")

    col_orig, col_heat = st.columns([1, 1])

    with col_orig:
        st.subheader("Original X-Ray")
        st.image(img_pil, use_container_width=True)

    with col_heat:
        selected = st.session_state.get("selected_heatmap", "Ensemble CAM")
        st.subheader(f"Heatmap View: {selected}")

        with st.spinner("Generating heatmap..."):
            if selected == "Ensemble CAM":
                try:
                    models_list = [densenet_model, resnet_model, effnet_model, convnext_model]
                    layers_list = [
                        densenet_model.features.norm5,
                        resnet_model.layer4[-1],
                        effnet_model.features[-1],
                        convnext_model.features[-1]
                    ]
                    tensors_list = [img_xrv, img_224, img_224, img_convnext]

                    heatmap_img = get_fused_heatmap(
                        models_list,
                        layers_list,
                        tensors_list,
                        img_pil,
                        heatmap_target
                    )

                    st.image(heatmap_img, use_container_width=True)
                    st.caption("Consensus map highlights the anatomical regions where the models agree on findings.")
                except Exception as e:
                    st.warning("Ensemble heatmap encountered an error rendering.")

            elif selected == "DenseNet121 XRV":
                heatmap_img = get_single_model_heatmap(
                    densenet_model,
                    densenet_model.features.norm5,
                    img_xrv,
                    img_pil,
                    1 if densenet_edema_prob >= 0.5 else 0
                )
                st.image(heatmap_img, use_container_width=True)
                st.caption("DenseNet121 XRV heatmap.")

            elif selected == "ResNet50":
                heatmap_img = get_single_model_heatmap(
                    resnet_model,
                    resnet_model.layer4[-1],
                    img_224,
                    img_pil,
                    1 if resnet_edema_prob >= 0.5 else 0
                )
                st.image(heatmap_img, use_container_width=True)
                st.caption("ResNet50 heatmap.")

            elif selected == "EfficientNet-B0":
                heatmap_img = get_single_model_heatmap(
                    effnet_model,
                    effnet_model.features[-1],
                    img_224,
                    img_pil,
                    1 if effnet_edema_prob >= 0.5 else 0
                )
                st.image(heatmap_img, use_container_width=True)
                st.caption("EfficientNet-B0 heatmap.")

            elif selected == "ConvNeXt-Tiny":
                try:
                    heatmap_img = get_single_model_heatmap(
                        convnext_model,
                        convnext_model.features[-1],
                        img_convnext,
                        img_pil,
                        1 if convnext_edema_prob >= 0.5 else 0
                    )
                    st.image(heatmap_img, use_container_width=True)
                    st.caption("ConvNeXt-Tiny heatmap.")
                except Exception as e:
                    st.warning("ConvNeXt heatmap is not available yet.")