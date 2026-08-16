import argparse
import os
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

def main():
    dnn_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DNN_models'))
    os.makedirs(dnn_models_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = dnn_models_dir

    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--agent", type=str, required=True, 
                        help="Specific model to run (e.g., 'resnet18', 'vgg16', 'lenet5') or 'all'.")
    parser.add_argument("--output_dir", type=str, default=dnn_models_dir, help="Directory to save ONNX models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    supported_models = ['lenet5', 'alexnet', 'vgg16', 'googlenet', 'resnet18', 'resnet50']
    
    if args.agent.lower() == 'all':
        models_to_run = supported_models
    else:
        models_to_run = [args.agent]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

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
