# 🧩 Masked Autoencoder (MAE) — Streamlit Demo

**Students:** 22F-3327 & 22F-8803  
**Course:** Generative AI — Assignment 02

## 📋 Overview

This Streamlit application demonstrates a **Masked Autoencoder (MAE)** trained on the TinyImageNet dataset. The MAE architecture follows the ViT (Vision Transformer) design:

- **Encoder:** ViT-Base (768-dim, 12 layers, 12 heads)  
- **Decoder:** ViT-Small (384-dim, 12 layers, 6 heads)
- **Image Size:** 224×224, **Patch Size:** 16×16
- **Masking Ratio:** 75% (configurable in the app)
- **Checkpoint Size:** ~2GB

## 🚀 Quick Start (Run Locally)

```bash
# Clone the repository
git clone <your-repo-url>
cd GEN_AI_A02

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Place your checkpoint file (mae_checkpoint.pth) in this directory

# Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🌐 Deployment Guide

Since the checkpoint is **~2GB**, it cannot be pushed directly to GitHub (100MB limit).  
Below are the recommended deployment options:

### Option 1: Deploy on Hugging Face Spaces (★ Recommended)

Hugging Face Spaces provides **free hosting** with **16GB RAM** — ideal for large models.

#### Steps:
1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Create a new Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Streamlit** as the SDK
   - Name it (e.g., `mae-demo`)

3. **Upload the checkpoint to Hugging Face:**
   - Create a **Model repository** on Hugging Face
   - Upload `mae_checkpoint.pth` there
   - Copy the direct download URL:  
     `https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/mae_checkpoint.pth`

4. **Set the URL in `app.py`:**
   ```python
   CHECKPOINT_DIRECT_URL = "https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/mae_checkpoint.pth"
   ```

5. **Push files to the Space:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/mae-demo
   cd mae-demo
   # Copy app.py, requirements.txt, .streamlit/ folder here
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. Your app will be live at `https://YOUR_USERNAME-mae-demo.hf.space` 🎉

---

### Option 2: Deploy on Streamlit Cloud (with Google Drive)

#### Steps:
1. **Upload checkpoint to Google Drive:**
   - Upload `mae_checkpoint.pth` to Google Drive
   - Right-click → **Share** → **Anyone with the link** → **Viewer**
   - Copy the link: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Extract the `FILE_ID` from the URL

2. **Set the ID in `app.py`:**
   ```python
   CHECKPOINT_GDRIVE_ID = "YOUR_FILE_ID_HERE"
   ```

3. **Push code to GitHub** (without the checkpoint):
   ```bash
   # Make sure .gitignore excludes mae_checkpoint.pth
   echo "mae_checkpoint.pth" >> .gitignore
   git init
   git add .
   git commit -m "MAE Streamlit app"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

4. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository and `app.py`
   - Deploy!

> ⚠️ **Note:** Streamlit Cloud free tier has limited RAM (~1GB). The ~2GB model may cause memory issues. Consider Option 1 (Hugging Face Spaces) for better reliability.

---

### Option 3: Deploy on Streamlit Cloud (with Git LFS)

If you prefer keeping everything in one repo:

```bash
# Install Git LFS
git lfs install

# Track the large checkpoint file
git lfs track "mae_checkpoint.pth"

# Add and commit
git add .gitattributes
git add mae_checkpoint.pth
git add .
git commit -m "Add MAE app with LFS checkpoint"
git push origin main
```

> ⚠️ GitHub LFS free tier is limited to **1GB storage** and **1GB bandwidth/month**.  
> This may not be sufficient for a ~2GB file. Use Option 1 or 2 instead.

---

## 📁 Project Structure

```
GEN_AI_A02/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── mae_checkpoint.pth          # Trained model (~2GB, download from Kaggle)
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── gen-ai-a2-22f-3327-22f-8803.ipynb  # Training notebook (Kaggle)
```

## 🎨 App Features

- **📸 Image Upload & Reconstruction:** Upload any image and see the full MAE pipeline
- **🎭 Adjustable Mask Ratio:** Control how much of the image is masked (10%–95%)
- **📊 Training History:** View training and validation loss curves
- **📖 How It Works:** Educational section explaining the MAE architecture
- **🎨 Premium Dark UI:** Glassmorphism design with smooth animations

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Deep Learning | PyTorch |
| Architecture | Vision Transformer (ViT) |
| Dataset | TinyImageNet (200 classes, 100K images) |
| Training | 30 epochs, Dual GPU T4, Mixed Precision |

## 📝 Quick Deployment Checklist

- [ ] Download `mae_checkpoint.pth` from Kaggle
- [ ] Upload checkpoint to Google Drive or Hugging Face Hub
- [ ] Set `CHECKPOINT_GDRIVE_ID` or `CHECKPOINT_DIRECT_URL` in `app.py`
- [ ] Push code to GitHub
- [ ] Deploy on Hugging Face Spaces or Streamlit Cloud
- [ ] Get your public URL! 🎉
