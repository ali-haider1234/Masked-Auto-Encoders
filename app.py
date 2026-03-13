"""
Masked Autoencoder (MAE) - Streamlit Web Application
=====================================================
Students: 22F-3327 & 22F-8803
Course: Generative AI - Assignment 02

This app demonstrates the Masked Autoencoder model trained on TinyImageNet.
Users can upload images and see the MAE reconstruction process:
  1. Original Image
  2. Masked Image (75% patches removed)
  3. MAE Reconstruction
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="MAE - Masked Autoencoder Demo",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Premium UI
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Card styling */
    .info-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .info-card h3 {
        color: #a78bfa;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .info-card p, .info-card li {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.25);
    }

    .metric-value {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Upload area */
    .upload-section {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: border-color 0.3s ease;
    }

    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.7);
    }

    /* Sidebar */
    .sidebar-info {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .sidebar-info h4 {
        color: #a78bfa;
        font-weight: 600;
    }

    .sidebar-info p {
        color: rgba(255, 255, 255, 0.65);
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Architecture (same as in the notebook)
# ============================================================

# Hyperparameters
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
MASK_RATIO = 0.75
KEEP_COUNT = int(NUM_PATCHES * (1 - MASK_RATIO))  # 49

# Encoder Config (ViT-Base)
ENC_EMBED_DIM = 768
ENC_DEPTH = 12
ENC_HEADS = 12

# Decoder Config (ViT-Small)
DEC_EMBED_DIM = 384
DEC_DEPTH = 12
DEC_HEADS = 6


def patchify(imgs):
    """imgs: (N, 3, H, W) -> x: (N, L, patch_size**2 * 3)"""
    p = PATCH_SIZE
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


def unpatchify(x):
    """x: (N, L, patch_size**2 * 3) -> imgs: (N, 3, H, W)"""
    p = PATCH_SIZE
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


def random_masking(x, mask_ratio):
    """Perform per-sample random masking by shuffling."""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x_visible, mask, ids_restore = random_masking(x, mask_ratio)
        for block in self.blocks:
            x_visible = block(x_visible)
        x_visible = self.norm(x_visible)
        return x_visible, mask, ids_restore


class Decoder(nn.Module):
    def __init__(self, num_patches=196, enc_embed_dim=768,
                 dec_embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.):
        super().__init__()
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dec_embed_dim, nhead=num_heads,
                dim_feedforward=int(dec_embed_dim * mlp_ratio),
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(dec_embed_dim)
        self.decoder_pred = nn.Linear(dec_embed_dim, PATCH_SIZE**2 * 3, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, enc_depth=12, enc_heads=12,
                 dec_embed_dim=384, dec_depth=12, dec_heads=6,
                 mlp_ratio=4.):
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=enc_depth, num_heads=enc_heads,
            mlp_ratio=mlp_ratio
        )
        num_patches = (img_size // patch_size) ** 2
        self.decoder = Decoder(
            num_patches=num_patches, enc_embed_dim=embed_dim,
            dec_embed_dim=dec_embed_dim, depth=dec_depth,
            num_heads=dec_heads, mlp_ratio=mlp_ratio
        )

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        return pred, mask


# ============================================================
# Model Loading & Caching
# ============================================================

CHECKPOINT_LOCAL_PATH = "mae_checkpoint.pth"


@st.cache_resource
def load_model():
    """Load the MAE model and checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder().to(device)

    if os.path.exists(CHECKPOINT_LOCAL_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_LOCAL_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Handle DataParallel state dict keys (remove 'module.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            # Get training history if available
            train_hist = checkpoint.get('train_loss_history', [])
            val_hist = checkpoint.get('val_loss_history', [])
            epoch = checkpoint.get('epoch', 0)

            return model, device, True, train_hist, val_hist, epoch
        except Exception as e:
            st.warning(f"⚠️ Error loading checkpoint: {e}. Using untrained model.")
            model.eval()
            return model, device, False, [], [], 0
    else:
        st.warning(
            "⚠️ **Checkpoint not found.** Please make sure `mae_checkpoint.pth` "
            "is uploaded to this Hugging Face Space."
        )
        model.eval()
        return model, device, False, [], [], 0


# ============================================================
# Image Processing
# ============================================================

def preprocess_image(image):
    """Preprocess a PIL image for the model."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def denormalize(tensor):
    """Denormalize a tensor for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def run_mae_inference(model, image_tensor, device, mask_ratio=0.75):
    """Run MAE inference: returns original, masked, and reconstructed images."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Forward pass
        pred, mask = model(image_tensor, mask_ratio)

        # Unpatchify reconstruction
        reconstructed = unpatchify(pred)

        # Create masked visualization
        original_patches = patchify(image_tensor)
        masked_patches = original_patches * (1 - mask.unsqueeze(-1))
        masked_img = unpatchify(masked_patches)

        # Create full reconstruction (visible patches from original + reconstructed masked patches)
        # For a blended view: show original visible + reconstructed masked
        full_recon_patches = original_patches * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
        full_recon = unpatchify(full_recon_patches)

    # Move to CPU and denormalize
    original_display = denormalize(image_tensor.cpu()[0]).permute(1, 2, 0).numpy()
    masked_display = denormalize(masked_img.cpu()[0]).permute(1, 2, 0).numpy()
    recon_display = denormalize(reconstructed.cpu()[0]).permute(1, 2, 0).numpy()
    full_recon_display = denormalize(full_recon.cpu()[0]).permute(1, 2, 0).numpy()

    mask_percentage = mask.cpu()[0].mean().item() * 100

    return original_display, masked_display, recon_display, full_recon_display, mask_percentage


def create_comparison_figure(original, masked, reconstructed, full_recon, mask_pct):
    """Create a side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    titles = [
        "🖼️ Original Image",
        f"🧩 Masked ({mask_pct:.0f}%)",
        "🔄 Raw Reconstruction",
        "✨ Blended Result"
    ]

    images = [original, masked, reconstructed, full_recon]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12,
                     color='white')
        ax.axis('off')

    fig.patch.set_facecolor('#0e1117')
    fig.tight_layout(pad=2.0)
    return fig


def create_loss_figure(train_hist, val_hist):
    """Create training loss plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs_range = range(1, len(train_hist) + 1)
    ax.plot(epochs_range, train_hist, 'o-', color='#667eea', linewidth=2,
            markersize=4, label='Train Loss', alpha=0.9)
    ax.plot(epochs_range, val_hist, 's-', color='#f59e0b', linewidth=2,
            markersize=4, label='Val Loss', alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=12, color='white')
    ax.set_ylabel('MSE Loss', fontsize=12, color='white')
    ax.set_title('Training & Validation Loss History', fontsize=14,
                 fontweight='bold', color='white')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2, color='white')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


# ============================================================
# Streamlit App Layout
# ============================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧩 Masked Autoencoder (MAE)</h1>
        <p>Self-Supervised Visual Representation Learning | ViT-Base Encoder + ViT-Small Decoder | TinyImageNet</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        mask_ratio = st.slider(
            "🎭 Mask Ratio",
            min_value=0.1, max_value=0.95, value=0.75, step=0.05,
            help="Percentage of patches to mask (default: 75%)"
        )

        st.markdown("---")

        st.markdown("""
        <div class="sidebar-info">
            <h4>📘 About MAE</h4>
            <p>Masked Autoencoders learn visual representations by reconstructing
            masked portions of input images. Only 25% of patches are visible
            to the encoder, making it highly efficient.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <h4>🏗️ Architecture</h4>
            <p>
            <strong>Encoder:</strong> ViT-Base (768d, 12L, 12H)<br>
            <strong>Decoder:</strong> ViT-Small (384d, 12L, 6H)<br>
            <strong>Image Size:</strong> 224×224<br>
            <strong>Patch Size:</strong> 16×16<br>
            <strong>Patches:</strong> 196 total, 49 visible
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <h4>👥 Students</h4>
            <p>22F-3327 & 22F-8803<br>
            Generative AI — Assignment 02</p>
        </div>
        """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Loading MAE model..."):
        model, device, model_loaded, train_hist, val_hist, trained_epochs = load_model()

    # Model Status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{"✅" if model_loaded else "⚠️"}</div>
            <div class="metric-label">Model Status</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{device.type.upper()}</div>
            <div class="metric-label">Device</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{trained_epochs}</div>
            <div class="metric-label">Trained Epochs</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mask_ratio*100:.0f}%</div>
            <div class="metric-label">Mask Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎨 **Image Reconstruction**", "📊 **Training History**", "📖 **How It Works**"])

    with tab1:
        st.markdown("### Upload an Image for MAE Reconstruction")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Upload any image to see the MAE reconstruction process"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            # Show original image
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("#### 📥 Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"Original size: {image.size[0]}×{image.size[1]}")

            with col_right:
                # Process
                with st.spinner("🔮 Running MAE inference..."):
                    image_tensor = preprocess_image(image)
                    original, masked, reconstructed, full_recon, mask_pct = run_mae_inference(
                        model, image_tensor, device, mask_ratio
                    )

                st.markdown("#### 🔬 MAE Reconstruction Pipeline")
                fig = create_comparison_figure(original, masked, reconstructed, full_recon, mask_pct)
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")

            # Detailed views
            st.markdown("### 🔍 Detailed View")
            det_col1, det_col2, det_col3, det_col4 = st.columns(4)

            with det_col1:
                st.markdown("**Original**")
                st.image(np.clip(original, 0, 1), use_container_width=True)

            with det_col2:
                st.markdown(f"**Masked ({mask_pct:.0f}%)**")
                st.image(np.clip(masked, 0, 1), use_container_width=True)

            with det_col3:
                st.markdown("**Raw Reconstruction**")
                st.image(np.clip(reconstructed, 0, 1), use_container_width=True)

            with det_col4:
                st.markdown("**Blended Result**")
                st.image(np.clip(full_recon, 0, 1), use_container_width=True)

        else:
            st.markdown("""
            <div class="upload-section">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">📸</p>
                <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem;">
                    Drag & drop an image here, or click to browse
                </p>
                <p style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">
                    Supported formats: PNG, JPG, JPEG, BMP, WebP
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        if train_hist and val_hist:
            st.markdown("### 📈 Training & Validation Loss")
            fig = create_loss_figure(train_hist, val_hist)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")

            # Loss summary
            lcol1, lcol2, lcol3, lcol4 = st.columns(4)
            with lcol1:
                st.metric("Initial Train Loss", f"{train_hist[0]:.4f}")
            with lcol2:
                st.metric("Final Train Loss", f"{train_hist[-1]:.4f}",
                          delta=f"{train_hist[-1] - train_hist[0]:.4f}")
            with lcol3:
                st.metric("Initial Val Loss", f"{val_hist[0]:.4f}")
            with lcol4:
                st.metric("Final Val Loss", f"{val_hist[-1]:.4f}",
                          delta=f"{val_hist[-1] - val_hist[0]:.4f}")

            # Epoch details table
            with st.expander("📋 Epoch-wise Loss Details"):
                import pandas as pd
                df = pd.DataFrame({
                    'Epoch': range(1, len(train_hist) + 1),
                    'Train Loss': [f"{x:.6f}" for x in train_hist],
                    'Val Loss': [f"{x:.6f}" for x in val_hist]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 Training history will be displayed when a trained checkpoint is loaded.")

    with tab3:
        st.markdown("### 🧠 How Masked Autoencoders Work")

        st.markdown("""
        <div class="info-card">
            <h3>1️⃣ Patchification</h3>
            <p>The input image (224×224) is divided into non-overlapping patches of size 16×16,
            producing a total of <strong>196 patches</strong>. Each patch is treated as a "token"
            for the Vision Transformer.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>2️⃣ Random Masking (75%)</h3>
            <p>A large portion of patches (75%) are randomly removed. Only <strong>49 visible patches</strong>
            (25%) are kept and sent to the encoder. This aggressive masking forces the model to learn
            meaningful visual representations.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>3️⃣ Encoder (ViT-Base)</h3>
            <p>The visible patches pass through a <strong>12-layer Vision Transformer</strong> with
            768-dimensional embeddings and 12 attention heads. The encoder only processes the 25%
            visible patches, making training very efficient.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>4️⃣ Decoder (ViT-Small)</h3>
            <p>The decoder receives the encoded visible patches plus learnable mask tokens for
            the missing 75%. It uses a <strong>12-layer transformer</strong> with 384-dimensional
            embeddings to reconstruct the full set of patches.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>5️⃣ Reconstruction Loss</h3>
            <p>MSE loss is computed <strong>only on masked patches</strong>, pushing the model
            to predict the missing visual content. This self-supervised approach eliminates
            the need for labeled data!</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🧩 <strong>Masked Autoencoder Demo</strong> — Generative AI Assignment 02</p>
        <p>Students: 22F-3327 & 22F-8803 | Built with ❤️ using Streamlit & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
