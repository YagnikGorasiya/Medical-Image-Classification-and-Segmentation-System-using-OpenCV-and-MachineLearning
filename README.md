# 🩺 Medical Image Classification and Segmentation System (OpenCV + ML)

A complete Computer Vision project with:
- **OpenCV** (Filtering, Feature Detection, Segmentation, Video Processing)
- **Deep Learning (PyTorch Transfer Learning - ResNet18)** for classification
- **FastAPI** backend API
- **Streamlit** UI frontend

This project supports 3 medical modules:
1. **Chest X-ray**: NORMAL vs PNEUMONIA
2. **Brain MRI**: NO_TUMOR vs TUMOR
3. **Skin Lesion**: BENIGN vs MALIGNANT

---

## ✅ Syllabus Coverage (CV)
- Image Filtering (Gaussian, Histogram Equalization)
- Feature Detection (Harris Corners)
- Image Segmentation (Threshold, Watershed)
- Image Classification (CNN Transfer Learning)
- Video Processing (Optical Flow - Lucas Kanade)
- Object/Motion Tracking (Optical Flow points)

---

## 🧱 Project Architecture
- **Backend (FastAPI)**: receives image/video -> runs OpenCV or CNN -> returns result
- **Frontend (Streamlit)**: upload image/video -> select module + task -> display output/prediction

---

## 📂 Folder Structure
CV_Project/
├── backend/
│ ├── app/
│ ├── models/
│ ├── requirements.txt
│ └── Dockerfile
├── frontend/
│ ├── app.py
│ ├── requirements.txt
│ └── Dockerfile
├── training/
│ ├── split_brain_dataset.py
│ ├── train_brain.py
│ ├── train_skin.py
│ ├── train_chest.py
│ └── (other helper scripts)
├── sample_data/
│ ├── chest/
│ ├── brain/
│ ├── skin/
│ ├── videos/
│ └── README.md
├── docker-compose.yml
├── .gitignore
└── README.md

---

## ⚙️ Requirements
- Python **3.10 / 3.11** recommended
- pip
- (Optional) Docker Desktop

> NOTE: If you faced "Numpy is not available" during training, install:
`numpy==1.26.4`.

---

## 🚀 Run Locally (Without Docker)

### 1) Create & activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

Install backend requirements
cd backend
pip install -r requirements.txt

Start backend
python -m uvicorn app.main:app --reload --port 8000

Test: http://127.0.0.1:8000/health
Install frontend requirements

Open a NEW terminal:

cd frontend
pip install -r requirements.txt
Start frontend
streamlit run app.py

Open: http://localhost:8501

Create:

datasets/
  chest_xray/
  brain_mri_raw/
  brain_mri/
  skin_lesion_raw/
  skin_lesion/

Brain MRI split (yes/no -> train/val/test)
python training/split_brain_dataset.py

Train Brain model
python training/train_brain.py
Output: backend/models/brain_cnn.pt

Train Skin model
python training/train_skin.py
Output: backend/models/skin_cnn.pt

Train Chest model
python training/train_chest.py
Output: backend/models/chest_cnn.pt

🧪 Testing using sample_data

See sample_data/README.md for how to fill it.
Then in Streamlit:
Choose module: chest/brain/skin
Choose mode: Classification / Processing / Segmentation / Feature
Upload images from sample_data folders

🐳 Run with Docker (Backend + Frontend together)
1) Install Docker Desktop
2) From project root:
docker-compose up --build

Open: 

Backend: http://127.0.0.1:8000/health

Frontend: http://localhost:8501

Stop:

docker-compose down

🌍 Deployment (Free)
Backend (FastAPI) -> Render

Build command: pip install -r requirements.txt

Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT

Frontend (Streamlit) -> Streamlit Community Cloud

App file: frontend/app.py

Set API_BASE_URL to your Render backend URL

👨‍💻 Author

Developed by: (Yagnik Gorasiya)
Computer Vision Project
