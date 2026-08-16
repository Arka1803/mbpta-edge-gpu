import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import gumbel_r
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Model Definitions
# -----------------------------------------------------------------------------
class LeNet5(nn.Module):
    """
    Standard LeNet-5 adapted for 3-channel 224x224 input to match the uniform
    preprocessing requested (ImageNet style).
    """
    def __init__(self, num_classes=1000):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 224 -> pool -> 112 -> conv(5) -> 108 -> pool -> 54
        self.classifier = nn.Sequential(
            nn.Linear(16 * 54 * 54, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_model(agent_name: str, device: torch.device) -> nn.Module:
    """Loads the specified pre-trained model."""
    name = agent_name.lower()
    
    if name == 'lenet5':
        model = LeNet5()
    elif name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif name == 'googlenet':
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    elif name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {agent_name}")
        
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 2. Block Maxima & Gumbel Fitting
# -----------------------------------------------------------------------------
def apply_block_maxima(data: np.ndarray, block_size: int = 50) -> np.ndarray:
    """Extracts block maxima from a 1D array of observations."""
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        return np.array([])
    blocks = data[:n_blocks * block_size].reshape((n_blocks, block_size))
    return np.max(blocks, axis=1)

def fit_gumbel_distribution(maxima: np.ndarray):
    """Fits the Gumbel distribution (Right-skewed) and returns params."""
    if len(maxima) == 0:
        return None, None
    # fit returns (loc, scale)
    loc, scale = gumbel_r.fit(maxima)
    return loc, scale

def plot_separate_distributions(scenes_data: list, dmp: float, out_dir: str, agent_name: str):
    """Plots fitted Gumbel PDF and CDF in separate figures for each scene."""
    if not scenes_data:
        return
        
    # Determine the global min and max x values for consistent plotting
    all_maxima = np.concatenate([data['maxima'] for data in scenes_data])
    global_min = min(all_maxima)
    global_max = max(all_maxima)
    
    # Expand range slightly for plotting
    global_scale = max([data['scale'] for data in scenes_data])
    x = np.linspace(global_min - 2 * global_scale, global_max + 5 * global_scale, 500)
    
    plt.rcParams.update({'font.size': 14}) # Increase font size

    for data in scenes_data:
        loc = data['loc']
        scale = data['scale']
        scene_name = data['scene_name']
        
        # Calculate pWCET for this scene based on DMP
        pwcet = gumbel_r.ppf(1 - dmp, loc=loc, scale=scale)
        label = f"{scene_name} (pWCET={pwcet:.2f})"
        
        # Plot PDF
        plt.figure(figsize=(10, 6))
        pdf = gumbel_r.pdf(x, loc=loc, scale=scale)
        plt.plot(x, pdf, lw=2, label=label, color='blue')
        plt.title(f"PDF: {agent_name.upper()} - {scene_name}", fontsize=16)
        plt.xlabel("Inference Time (ms)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.xlim([global_min - 2 * global_scale, global_max + 5 * global_scale])
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"{agent_name}_{scene_name}_pdf.png")
        plt.savefig(pdf_path, dpi=300)
        plt.close()
        
        # Plot CDF
        plt.figure(figsize=(10, 6))
        cdf = gumbel_r.cdf(x, loc=loc, scale=scale)
        plt.plot(x, cdf, lw=2, label=scene_name, color='green')
        plt.title(f"CDF: {agent_name.upper()} - {scene_name}", fontsize=16)
        plt.xlabel("Inference Time (ms)", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        plt.xlim([global_min - 2 * global_scale, global_max + 5 * global_scale])
        plt.axhline(y=1-dmp, color='red', linestyle='--', label=f'Threshold (1-DMP={1-dmp})')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        cdf_path = os.path.join(out_dir, f"{agent_name}_{scene_name}_cdf.png")
        plt.savefig(cdf_path, dpi=300)
        plt.close()


# -----------------------------------------------------------------------------
# 3. Profiling Logic
# -----------------------------------------------------------------------------
def profile_video(video_path: str, model: nn.Module, device: torch.device, transform) -> list:
    """Profiles frame-by-frame inference time for a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return []
        
    inference_times = []
    
    # Pre-allocate CUDA events if using GPU
    use_cuda = device.type == 'cuda'
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Apply standard ImageNet preprocessing
        input_tensor = transform(pil_img)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        if use_cuda:
            start_event.record()
            with torch.no_grad():
                _ = model(input_batch)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(input_batch)
            elapsed_time_ms = (time.perf_counter() - t0) * 1000.0
            
        inference_times.append(elapsed_time_ms)
        
    cap.release()
    return inference_times

def run_profiling(agent_name: str, scenes_dir: str, results_dir: str, dmp: float):
    """Main execution pipeline for a specific model."""
    print(f"--- Starting profiling for agent: {agent_name} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Warning: CUDA is not available. Using CPU. Timing events will still work but won't reflect GPU execution.")
    
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    try:
        model = load_model(agent_name, device)
    except Exception as e:
        print(f"Failed to load model {agent_name}: {e}")
        return

    # Find all scene subfolders
    if not os.path.exists(scenes_dir):
        print(f"Scenes directory '{scenes_dir}' not found.")
        return
        
    scene_folders = [f.path for f in os.scandir(scenes_dir) if f.is_dir()]
    
    scenes_data = []
    
    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        print(f"  Processing scene: {scene_name}")
        
        video_files = glob.glob(os.path.join(scene_folder, "*.mp4")) + \
                      glob.glob(os.path.join(scene_folder, "*.avi"))
                      
        if not video_files:
            print(f"    No videos found in {scene_folder}")
            continue
            
        all_times = []
        for video_file in video_files:
            times = profile_video(video_file, model, device, transform)
            all_times.extend(times)
            
        if not all_times:
            continue
            
        times_array = np.array(all_times)
        
        # Statistical Analysis
        maxima = apply_block_maxima(times_array, block_size=50)
        loc, scale = fit_gumbel_distribution(maxima)
        
        if loc is None:
            print(f"    Not enough frames in {scene_name} for block size 50.")
            continue
            
        print(f"    -> Fitted Gumbel: loc={loc:.2f} ms, scale={scale:.2f} ms")
        
        # Save Outputs
        # Save to CSV
        csv_dir = os.path.join(results_dir, "csv_files")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{agent_name}_{scene_name}_raw.csv")
        df = pd.DataFrame({'inference_time_ms': times_array})
        df.to_csv(csv_path, index=False)
        
        # Save to MAT (raw times, maxima, fitted parameters)
        mat_dir = os.path.join(results_dir, "mat_files")
        os.makedirs(mat_dir, exist_ok=True)
        mat_path = os.path.join(mat_dir, f"{agent_name}_{scene_name}.mat")
        mat_data = {
            'inference_times': times_array,
            'block_maxima': maxima,
            'gumbel_loc': loc,
            'gumbel_scale': scale,
            'model_name': agent_name,
            'scene_name': scene_name
        }
        sio.savemat(mat_path, mat_data)
        
        # Collect for Plot
        scenes_data.append({
            'maxima': maxima,
            'loc': loc,
            'scale': scale,
            'scene_name': scene_name
        })
        
        print(f"    -> Exported results to {csv_path}, and {mat_path}")
        
    if scenes_data:
        plot_dir = os.path.join(results_dir, "plots_png")
        os.makedirs(plot_dir, exist_ok=True)
        plot_separate_distributions(scenes_data, dmp, plot_dir, agent_name)
        print(f"  -> Generated PDF and CDF plots in: {plot_dir}")

# -----------------------------------------------------------------------------
# 4. Entry Point
# -----------------------------------------------------------------------------
def main():
    # Set TORCH_HOME to download models into local DNN_models directory
    dnn_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DNN_models'))
    os.makedirs(dnn_models_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = dnn_models_dir

    parser = argparse.ArgumentParser(description="DNN GPU Inference Profiler for WCET Analysis")
    parser.add_argument("--agent", type=str, required=True, 
                        help="Specific model to run (e.g., 'resnet18', 'vgg16', 'lenet5') or 'all'.")
    parser.add_argument("--dmp", type=float, default=1e-6,
                        help="Deadline Miss Probability (default: 1e-6)")
    
    args = parser.parse_args()
    
    # Configuration
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scenes'))
    results_dir = "results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    supported_models = ['lenet5', 'alexnet', 'vgg16', 'googlenet', 'resnet18', 'resnet50']
    
    if args.agent.lower() == 'all':
        models_to_run = supported_models
    else:
        models_to_run = [args.agent]
        
    for model_name in models_to_run:
        if model_name.lower() not in supported_models:
            print(f"Warning: {model_name} is not in the standard supported list. Attempting to load anyway.")
        run_profiling(model_name, scenes_dir, results_dir, args.dmp)
        
    print("All tasks completed.")

if __name__ == "__main__":
    main()
