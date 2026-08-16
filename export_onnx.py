import argparse
import os

# Set TORCH_HOME before importing torchvision so that all weight downloads
# and cache files land in ../DNN_models/ regardless of system defaults.
_DNN_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DNN_models'))
os.makedirs(_DNN_MODELS_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = _DNN_MODELS_DIR

import torch
import torch.nn as nn
import torchvision.models as models

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

# Mapping of model name -> (constructor, weights enum)
_MODEL_REGISTRY = {
    'alexnet':   (models.alexnet,    models.AlexNet_Weights.DEFAULT),
    'vgg16':     (models.vgg16,      models.VGG16_Weights.DEFAULT),
    'googlenet': (models.googlenet,  models.GoogLeNet_Weights.DEFAULT),
    'resnet18':  (models.resnet18,   models.ResNet18_Weights.DEFAULT),
    'resnet50':  (models.resnet50,   models.ResNet50_Weights.DEFAULT),
}

def download_weights(model_names: list):
    """
    Explicitly pre-downloads all pretrained weights for the given model names
    before any ONNX conversion begins. LeNet-5 has no pretrained weights.
    """
    print("--- Pre-downloading pretrained weights ---")
    for name in model_names:
        if name == 'lenet5':
            print(f"  [{name}] No pretrained weights (custom architecture) — skipping.")
            continue
        if name not in _MODEL_REGISTRY:
            print(f"  [{name}] Unknown model — skipping download.")
            continue
        constructor, weights_enum = _MODEL_REGISTRY[name]
        print(f"  [{name}] Downloading / verifying weights -> {_DNN_MODELS_DIR} ...")
        try:
            constructor(weights=weights_enum)   # triggers download if not cached
            print(f"  [{name}] OK")
        except Exception as e:
            print(f"  [{name}] WARNING: download failed: {e}")
    print("--- Weights ready ---\n")

def load_model(agent_name: str, device: torch.device) -> nn.Module:
    """Loads the specified pre-trained model (weights already cached)."""
    name = agent_name.lower()

    if name == 'lenet5':
        model = LeNet5()
    elif name in _MODEL_REGISTRY:
        constructor, weights_enum = _MODEL_REGISTRY[name]
        model = constructor(weights=weights_enum)
    else:
        raise ValueError(f"Unsupported model: {agent_name}")

    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--agent", type=str, required=True,
                        help="Specific model to run (e.g., 'resnet18', 'vgg16', 'lenet5') or 'all'.")
    parser.add_argument("--output_dir", type=str, default=_DNN_MODELS_DIR, help="Directory to save ONNX models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    supported_models = ['lenet5', 'alexnet', 'vgg16', 'googlenet', 'resnet18', 'resnet50']
    
    if args.agent.lower() == 'all':
        models_to_run = supported_models
    else:
        models_to_run = [args.agent]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Step 1: Download / verify all pretrained weights before any ONNX work
    download_weights(models_to_run)

    for model_name in models_to_run:
        print(f"Exporting {model_name} to ONNX...")
        try:
            model = load_model(model_name, device)
            onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path, 
                export_params=True, 
                opset_version=11, 
                do_constant_folding=True,
                input_names=['input'], 
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"Successfully exported {model_name} to {onnx_path}")
        except Exception as e:
            print(f"Failed to export {model_name}: {e}")

if __name__ == "__main__":
    main()
