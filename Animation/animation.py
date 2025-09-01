#!/usr/bin/env python3
"""
Stage Transition VAE with Command Line Interface - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import argparse
import glob
import gradio as gr
import tempfile
from pathlib import Path

class VAE(nn.Module):
    def __init__(self, latent_dim=64, image_size=64, num_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size
        self.encoder_output_size = 256 * (image_size // 16) * (image_size // 16)
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, image_size // 16, image_size // 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ConditionalVAE(VAE):
    def __init__(self, latent_dim=64, image_size=64, num_channels=3, num_classes=3):
        super(ConditionalVAE, self).__init__(latent_dim, image_size, num_channels)
        self.num_classes = num_classes
        
        # Modify encoder to include class information
        self.class_embedding = nn.Embedding(num_classes, 16)
        
        # Modify latent space layers to include class information
        self.fc_mu = nn.Linear(self.encoder_output_size + 16, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size + 16, latent_dim)
        
        # Modify decoder input to include class information
        self.decoder_input = nn.Linear(latent_dim + 16, self.encoder_output_size)
    
    def encode(self, x, labels):
        x_encoded = self.encoder(x)
        class_emb = self.class_embedding(labels)
        x_combined = torch.cat([x_encoded, class_emb], dim=1)
        mu = self.fc_mu(x_combined)
        logvar = self.fc_logvar(x_combined)
        return mu, logvar
    
    def decode(self, z, labels):
        class_emb = self.class_embedding(labels)
        z_combined = torch.cat([z, class_emb], dim=1)
        x = self.decoder_input(z_combined)
        x = self.decoder(x)
        return x
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

class StageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.encoded_labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

def load_data(stage_dirs, image_size=64):
    """Load images from three stage directories"""
    image_paths = []
    labels = []
    
    for stage_idx, stage_dir in enumerate(stage_dirs):
        stage_name = f"stage_{stage_idx}"
        # Support multiple image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        for extension in image_extensions:
            for img_file in glob.glob(os.path.join(stage_dir, extension)):
                image_paths.append(img_file)
                labels.append(stage_name)
    
    if not image_paths:
        raise ValueError(f"No images found in the specified directories: {stage_dirs}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return StageDataset(image_paths, labels, transform)

def train_vae(model, dataloader, epochs=50, device='cuda'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, ConditionalVAE):
                recon_batch, mu, logvar = model(data, labels)
            else:
                recon_batch, mu, logvar = model(data)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(dataloader.dataset):.4f}, '
              f'Recon Loss: {recon_loss_total/len(dataloader.dataset):.4f}, '
              f'KL Loss: {kl_loss_total/len(dataloader.dataset):.4f}')
    
    return model

def spherical_interpolation(z1, z2, t):
    """Spherical linear interpolation for smoother transitions"""
    z1_norm = torch.norm(z1)
    z2_norm = torch.norm(z2)
    
    omega = torch.acos(torch.dot(z1/z1_norm, z2/z2_norm))
    sin_omega = torch.sin(omega)
    
    if sin_omega < 1e-10:
        return (1 - t) * z1 + t * z2
    
    return (torch.sin((1 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2

def generate_interpolation_sequence(model, dataloader, num_frames=30, device='cuda'):
    """Generate interpolation sequence between all stages"""
    model.eval()
    
    # Get representative latent vectors for each stage
    stage_vectors = {}
    
    with torch.no_grad():
        for stage_idx in range(3):
            # Find an image from this stage
            for data, labels in dataloader:
                if labels.item() == stage_idx:
                    data = data.to(device)
                    labels = labels.to(device)
                    if isinstance(model, ConditionalVAE):
                        mu, _ = model.encode(data, labels)
                    else:
                        mu, _ = model.encode(data)
                    stage_vectors[stage_idx] = mu.squeeze()
                    break
    
    # Generate interpolation sequences
    sequences = {}
    
    # Stage 0 -> Stage 1
    z0, z1 = stage_vectors[0], stage_vectors[1]
    seq_0_1 = []
    for t in np.linspace(0, 1, num_frames):
        z_interp = spherical_interpolation(z0, z1, torch.tensor(t, device=device))
        if isinstance(model, ConditionalVAE):
            # Interpolate label between 0 and 1
            label_val = t
            label = torch.tensor([int(round(label_val))], device=device)
            recon = model.decode(z_interp.unsqueeze(0), label)
        else:
            recon = model.decode(z_interp.unsqueeze(0))
        seq_0_1.append(recon.squeeze().cpu().detach())  # Added .detach()
    
    # Stage 1 -> Stage 2
    z1, z2 = stage_vectors[1], stage_vectors[2]
    seq_1_2 = []
    for t in np.linspace(0, 1, num_frames):
        z_interp = spherical_interpolation(z1, z2, torch.tensor(t, device=device))
        if isinstance(model, ConditionalVAE):
            # Interpolate label between 1 and 2
            label_val = 1 + t
            label = torch.tensor([int(round(label_val))], device=device)
            recon = model.decode(z_interp.unsqueeze(0), label)
        else:
            recon = model.decode(z_interp.unsqueeze(0))
        seq_1_2.append(recon.squeeze().cpu().detach())  # Added .detach()
    
    # Full sequence: 0->1->2
    full_sequence = seq_0_1 + seq_1_2
    
    return full_sequence, seq_0_1, seq_1_2

def create_animation(frames, output_path, fps=10):
    """Create animation from generated frames"""
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    
    # Convert frames to numpy and denormalize
    frames_np = [frame.permute(1, 2, 0).numpy() for frame in frames]
    
    # Create animation
    ims = []
    for i, frame in enumerate(frames_np):
        im = ax.imshow(frame, animated=True)
        title = ax.text(0.5, 1.05, f'Frame {i+1}/{len(frames)}', 
                       transform=ax.transAxes, ha='center')
        ims.append([im, title])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved to {output_path}")

def generate_specific_state(model, dataloader, stage_idx, interpolation_factor=0.5, device='cuda'):
    """Generate a specific state between stages"""
    model.eval()
    
    # Get latent vectors for adjacent stages
    stage_vectors = {}
    
    with torch.no_grad():
        for target_idx in [stage_idx, stage_idx + 1]:
            for data, labels in dataloader:
                if labels.item() == target_idx:
                    data = data.to(device)
                    labels = labels.to(device)
                    if isinstance(model, ConditionalVAE):
                        mu, _ = model.encode(data, labels)
                    else:
                        mu, _ = model.encode(data)
                    stage_vectors[target_idx] = mu.squeeze()
                    break
    
    if len(stage_vectors) < 2:
        raise ValueError(f"Could not find images for stages {stage_idx} and {stage_idx + 1}")
    
    # Interpolate
    z1, z2 = stage_vectors[stage_idx], stage_vectors[stage_idx + 1]
    z_interp = spherical_interpolation(z1, z2, torch.tensor(interpolation_factor, device=device))
    
    if isinstance(model, ConditionalVAE):
        # Interpolate label
        label_val = stage_idx + interpolation_factor
        label = torch.tensor([int(round(label_val))], device=device)
        result = model.decode(z_interp.unsqueeze(0), label)
    else:
        result = model.decode(z_interp.unsqueeze(0))
    
    return result.squeeze().cpu().detach()  # Added .detach()

class StageAnimationApp:
    def __init__(self, stage_dirs, image_size=64, use_conditional=True, latent_dim=64):
        self.stage_dirs = stage_dirs
        self.image_size = image_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize model
        if use_conditional:
            self.model = ConditionalVAE(latent_dim=latent_dim, image_size=image_size, num_classes=3)
        else:
            self.model = VAE(latent_dim=latent_dim, image_size=image_size)
        
        self.model = self.model.to(self.device)
        self.is_trained = False
        self.dataset = None
    
    def load_data(self):
        """Load and prepare data"""
        self.dataset = load_data(self.stage_dirs, self.image_size)
        print(f"Loaded {len(self.dataset)} images from {len(self.stage_dirs)} stages")
        
        # Count images per stage
        stage_counts = {}
        for _, label in self.dataset:
            stage_name = self.dataset.label_encoder.inverse_transform([label])[0]
            stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        
        for stage, count in stage_counts.items():
            print(f"  {stage}: {count} images")
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the VAE model"""
        if self.dataset is None:
            self.load_data()
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        print("Training VAE model...")
        self.model = train_vae(self.model, dataloader, epochs, self.device)
        self.is_trained = True
        print("Training completed!")
    
    def generate_animation(self, output_path="animation.gif", num_frames=30):
        """Generate full animation"""
        if not self.is_trained:
            print("Please train the model first!")
            return
        
        if self.dataset is None:
            self.load_data()
        
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        print("Generating animation...")
        full_sequence, _, _ = generate_interpolation_sequence(
            self.model, dataloader, num_frames, self.device
        )
        create_animation(full_sequence, output_path, fps=15)
    
    def generate_specific_frame(self, stage, progress):
        """Generate a specific frame between stages"""
        if not self.is_trained:
            print("Please train the model first!")
            return None
        
        if self.dataset is None:
            self.load_data()
        
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        stage_idx = min(max(int(stage), 0), 1)  # 0 or 1 (between stage0-1 or stage1-2)
        interpolation_factor = min(max(progress, 0.0), 1.0)
        
        try:
            result = generate_specific_state(
                self.model, dataloader, stage_idx, interpolation_factor, self.device
            )
            return result
        except ValueError as e:
            print(f"Error generating frame: {e}")
            return None
    
    def save_model(self, path="stage_vae.pth"):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'image_size': self.image_size,
            'is_conditional': isinstance(self.model, ConditionalVAE),
            'latent_dim': self.model.latent_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="stage_vae.pth"):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.image_size = checkpoint['image_size']
        latent_dim = checkpoint.get('latent_dim', 64)
        
        if checkpoint['is_conditional']:
            self.model = ConditionalVAE(latent_dim=latent_dim, image_size=self.image_size)
        else:
            self.model = VAE(latent_dim=latent_dim, image_size=self.image_size)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.is_trained = True
        print(f"Model loaded from {path}")
    
    def show_sample_frames(self):
        """Show sample frames from the dataset"""
        if self.dataset is None:
            self.load_data()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        stage_samples = {0: None, 1: None, 2: None}
        
        # Get one sample from each stage
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            if stage_samples[label.item()] is None:
                stage_samples[label.item()] = img
            if all(sample is not None for sample in stage_samples.values()):
                break
        
        for i, (stage, img) in enumerate(stage_samples.items()):
            if img is not None:
                # Use .detach() to avoid gradient issues
                axes[i].imshow(img.permute(1, 2, 0).detach().numpy())
                axes[i].set_title(f"Stage {stage}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def create_interactive_interface(app):
    """Create an interactive Gradio interface for animation control"""
    
    def generate_single_frame(stage_transition, progress):
        """Generate a single frame based on user input"""
        if not app.is_trained:
            return None, "Please train the model first!"
        
        # Map stage transition to stage index
        stage_map = {"Stage 0 → 1": 0, "Stage 1 → 2": 1}
        stage_idx = stage_map[stage_transition]
        
        result = app.generate_specific_frame(stage_idx, progress)
        if result is not None:
            # Convert to PIL Image
            result_np = result.permute(1, 2, 0).detach().numpy()
            result_np = (result_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(result_np)
            return pil_image, f"Generated: {stage_transition} at {progress*100:.1f}%"
        return None, "Error generating frame"
    
    def generate_animation_custom(stage_transition, num_frames, fps):
        """Generate custom animation for a specific transition"""
        if not app.is_trained:
            return None, "Please train the model first!"
        
        stage_map = {"Stage 0 → 1": 0, "Stage 1 → 2": 1, "Full Sequence (0→1→2)": "full"}
        
        if app.dataset is None:
            app.load_data()
        
        dataloader = DataLoader(app.dataset, batch_size=1, shuffle=True)
        
        if stage_map[stage_transition] == "full":
            # Full sequence
            full_sequence, _, _ = generate_interpolation_sequence(
                app.model, dataloader, num_frames, app.device
            )
            frames = full_sequence
        else:
            # Single transition
            stage_idx = stage_map[stage_transition]
            z0, z1 = get_stage_vectors(app.model, dataloader, stage_idx, stage_idx + 1, app.device)
            frames = generate_transition_frames(app.model, z0, z1, num_frames, app.device)
        
        # Create animation
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            temp_path = tmp.name
        
        create_animation(frames, temp_path, fps)
        return temp_path, f"Animation created with {num_frames} frames at {fps} FPS"
    
    def get_stage_vectors(model, dataloader, stage_idx1, stage_idx2, device):
        """Get latent vectors for two stages"""
        stage_vectors = {}
        with torch.no_grad():
            for target_idx in [stage_idx1, stage_idx2]:
                for data, labels in dataloader:
                    if labels.item() == target_idx:
                        data = data.to(device)
                        labels = labels.to(device)
                        if isinstance(model, ConditionalVAE):
                            mu, _ = model.encode(data, labels)
                        else:
                            mu, _ = model.encode(data)
                        stage_vectors[target_idx] = mu.squeeze()
                        break
        return stage_vectors[stage_idx1], stage_vectors[stage_idx2]
    
    def generate_transition_frames(model, z0, z1, num_frames, device):
        """Generate frames for a single transition"""
        frames = []
        for t in np.linspace(0, 1, num_frames):
            z_interp = spherical_interpolation(z0, z1, torch.tensor(t, device=device))
            if isinstance(model, ConditionalVAE):
                label_val = t if torch.equal(z0, get_stage_vectors(model, 
                    DataLoader(app.dataset, batch_size=1, shuffle=True), 0, 1, device)[0]) else 1 + t
                label = torch.tensor([int(round(label_val))], device=device)
                recon = model.decode(z_interp.unsqueeze(0), label)
            else:
                recon = model.decode(z_interp.unsqueeze(0))
            frames.append(recon.squeeze().cpu().detach())
        return frames
    
    def generate_full_animation_custom(num_frames_per_transition, fps):
        """Generate full animation with custom settings"""
        if not app.is_trained:
            return None, "Please train the model first!"
        
        if app.dataset is None:
            app.load_data()
        
        dataloader = DataLoader(app.dataset, batch_size=1, shuffle=True)
        full_sequence, _, _ = generate_interpolation_sequence(
            app.model, dataloader, num_frames_per_transition, app.device
        )
        
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            temp_path = tmp.name
        
        create_animation(full_sequence, temp_path, fps)
        return temp_path, f"Full animation created with {num_frames_per_transition*2} frames at {fps} FPS"
    
    # Create the interface
    with gr.Blocks(title="Stage Transition Animator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Stage Transition Animation Generator")
        gr.Markdown("Control and generate smooth animations between different stages")
        
        with gr.Tab("Single Frame Generator"):
            gr.Markdown("## Generate a Specific Intermediate Frame")
            with gr.Row():
                stage_dropdown = gr.Dropdown(
                    choices=["Stage 0 → 1", "Stage 1 → 2"],
                    value="Stage 0 → 1",
                    label="Select Transition"
                )
                progress_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.01,
                    label="Progress (0.0 - 1.0)"
                )
            
            generate_btn = gr.Button("Generate Frame", variant="primary")
            frame_output = gr.Image(label="Generated Frame", height=300)
            status_text = gr.Textbox(label="Status")
            
            generate_btn.click(
                generate_single_frame,
                inputs=[stage_dropdown, progress_slider],
                outputs=[frame_output, status_text]
            )
        
        with gr.Tab("Custom Animation"):
            gr.Markdown("## Generate Custom Animations")
            with gr.Row():
                anim_type = gr.Dropdown(
                    choices=["Stage 0 → 1", "Stage 1 → 2", "Full Sequence (0→1→2)"],
                    value="Full Sequence (0→1→2)",
                    label="Animation Type"
                )
                num_frames = gr.Slider(
                    minimum=5, maximum=100, value=20, step=1,
                    label="Number of Frames per Transition"
                )
                fps_control = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Frames per Second"
                )
            
            generate_anim_btn = gr.Button("Generate Animation", variant="primary")
            anim_output = gr.Video(label="Generated Animation")
            anim_status = gr.Textbox(label="Status")
            
            generate_anim_btn.click(
                generate_animation_custom,
                inputs=[anim_type, num_frames, fps_control],
                outputs=[anim_output, anim_status]
            )
        
        with gr.Tab("Batch Generation"):
            gr.Markdown("## Generate Multiple Intermediate States")
            with gr.Row():
                num_steps = gr.Slider(
                    minimum=3, maximum=20, value=5, step=1,
                    label="Number of Intermediate Steps"
                )
                transition_select = gr.Dropdown(
                    choices=["Stage 0 → 1", "Stage 1 → 2", "Both Transitions"],
                    value="Both Transitions",
                    label="Select Transitions"
                )
            
            generate_batch_btn = gr.Button("Generate Batch", variant="primary")
            batch_gallery = gr.Gallery(
                label="Intermediate States",
                columns=5,
                height="auto"
            )
            batch_status = gr.Textbox(label="Status")
            
            def generate_batch_steps(num_steps, transition):
                if not app.is_trained:
                    return [], "Please train the model first!"
                
                transitions = []
                if transition == "Stage 0 → 1" or transition == "Both Transitions":
                    transitions.append(0)
                if transition == "Stage 1 → 2" or transition == "Both Transitions":
                    transitions.append(1)
                
                images = []
                for stage_idx in transitions:
                    for progress in np.linspace(0, 1, num_steps):
                        frame = app.generate_specific_frame(stage_idx, progress)
                        if frame is not None:
                            result_np = frame.permute(1, 2, 0).detach().numpy()
                            result_np = (result_np * 255).astype(np.uint8)
                            images.append(Image.fromarray(result_np))
                
                return images, f"Generated {len(images)} intermediate states"
            
            generate_batch_btn.click(
                generate_batch_steps,
                inputs=[num_steps, transition_select],
                outputs=[batch_gallery, batch_status]
            )
        
        with gr.Tab("Advanced Settings"):
            gr.Markdown("## Advanced Animation Settings")
            with gr.Row():
                custom_fps = gr.Slider(
                    minimum=1, maximum=60, value=15, step=1,
                    label="Frames per Second"
                )
                custom_frames = gr.Slider(
                    minimum=10, maximum=200, value=30, step=5,
                    label="Total Frames"
                )
            
            advanced_btn = gr.Button("Generate High-Quality Animation", variant="primary")
            advanced_output = gr.Video(label="High-Quality Animation")
            advanced_status = gr.Textbox(label="Status")
            
            advanced_btn.click(
                generate_full_animation_custom,
                inputs=[custom_frames, custom_fps],
                outputs=[advanced_output, advanced_status]
            )
        
        gr.Markdown("---")
        gr.Markdown("### 💡 Tips:")
        gr.Markdown("- Use higher FPS for smoother animations")
        gr.Markdown("- More frames = smoother transitions but larger file size")
        gr.Markdown("- Experiment with different progress values to see specific intermediate states")
    
    return demo

# Add this function to your existing code
def create_animation(frames, output_path, fps=10):
    """Create animation from generated frames - Enhanced version"""
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    
    # Convert frames to numpy and denormalize
    frames_np = [frame.permute(1, 2, 0).detach().numpy() for frame in frames]
    
    # Create animation
    ims = []
    for i, frame in enumerate(frames_np):
        im = ax.imshow(frame, animated=True)
        title = ax.text(0.5, 1.05, f'Frame {i+1}/{len(frames)}', 
                       transform=ax.transAxes, ha='center', fontsize=12, color='white',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
        ims.append([im, title])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    
    # Save with better quality
    ani.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    
    print(f"Animation saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Stage Transition VAE with Web Interface")
    parser.add_argument("--stage_dirs", nargs=3, required=True, 
                       help="Paths to the three stage directories")
    parser.add_argument("--image_size", type=int, default=64,
                       help="Image size for training")
    parser.add_argument("--latent_dim", type=int, default=64,
                       help="Latent dimension size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--conditional", action="store_true",
                       help="Use conditional VAE")
    parser.add_argument("--train", action="store_true",
                       help="Train the model")
    parser.add_argument("--model_path", default="stage_vae.pth",
                       help="Path to save/load model")
    parser.add_argument("--load_model", action="store_true",
                       help="Load pre-trained model")
    parser.add_argument("--launch_ui", action="store_true",
                       help="Launch web interface after training")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for web interface")
    
    args = parser.parse_args()
    
    # Create app
    app = StageAnimationApp(
        args.stage_dirs, 
        image_size=args.image_size,
        use_conditional=args.conditional,
        latent_dim=args.latent_dim
    )
    
    # Load model if specified
    if args.load_model and os.path.exists(args.model_path):
        app.load_model(args.model_path)
    
    # Train model if specified
    if args.train:
        app.train_model(epochs=args.epochs, batch_size=args.batch_size)
        app.save_model(args.model_path)
    
    # Launch web interface if requested
    if args.launch_ui:
        if not app.is_trained:
            print("Model not trained. Please train first or use --load_model with a trained model.")
            return
        
        print("Launching web interface...")
        print(f"Open http://localhost:{args.port} in your browser")
        
        demo = create_interactive_interface(app)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
    
    else:
        # Show sample frames and generate basic animation
        app.show_sample_frames()
        
        if app.is_trained:
            print("Generating sample animation...")
            app.generate_animation("sample_animation.gif")

if __name__ == "__main__":
    main()