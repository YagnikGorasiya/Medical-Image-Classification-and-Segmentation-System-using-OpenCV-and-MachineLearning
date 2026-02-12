from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from app.utils.io import bytes_to_bgr, bgr_to_png_bytes

from app.cv import enhancement, filtering_edges, features, segmentation, video_motion
from app.ml.model_registry import MODELS
from app.ml.cnn_classifier import predict_resnet18

app = FastAPI(title="Medical Image Classification & Segmentation System")

# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "Backend running successfully!"}


# -------------------------------------------------------
# IMAGE PROCESSING (OpenCV)
# -------------------------------------------------------
@app.post("/process/image")
async def process_image(
    file: UploadFile = File(...),
    task: str = Form(...)
):
    img_bgr = bytes_to_bgr(await file.read())

    if task == "hist_eq":
        out = enhancement.histogram_equalization_gray(img_bgr)

    elif task == "gaussian":
        out = filtering_edges.gaussian_blur(img_bgr)

    elif task == "canny":
        out = filtering_edges.canny(img_bgr)

    elif task == "harris":
        out = features.harris_corners(img_bgr)

    elif task == "threshold":
        out = segmentation.threshold_segment(img_bgr)

    elif task == "watershed":
        out = segmentation.watershed_segment(img_bgr)

    else:
        return JSONResponse({"error": "Invalid task"}, status_code=400)

    return Response(content=bgr_to_png_bytes(out), media_type="image/png")


# -------------------------------------------------------
# CNN PREDICTION (MULTI MODULE)
# -------------------------------------------------------
@app.post("/predict/cnn")
async def predict_cnn(
    file: UploadFile = File(...),
    module: str = Form(...)
):
    img_bgr = bytes_to_bgr(await file.read())

    if module not in MODELS:
        return JSONResponse({"error": "Invalid module selected"}, status_code=400)

    model_info = MODELS[module]
    model_path = model_info["path"]
    classes = model_info["classes"]

    prediction = predict_resnet18(img_bgr, model_path, classes)
    return prediction


# -------------------------------------------------------
# VIDEO OPTICAL FLOW
# -------------------------------------------------------
@app.post("/video/optical-flow")
async def run_optical_flow(file: UploadFile = File(...)):
    data = await file.read()
    temp_path = "temp_video.mp4"

    with open(temp_path, "wb") as f:
        f.write(data)

    result = video_motion.optical_flow_lk(temp_path)
    return result
