import torch
import torch.nn as nn

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.outputs = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def backward_hook_fn(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        def forward_hook_fn(module, input, output):
            self.outputs = output

        self.target_layer.register_backward_hook(backward_hook_fn)
        self.target_layer.register_forward_hook(forward_hook_fn)

    def generate_cam(self, input_tensor, target_category=None):
        # Forward pass
        model_output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(model_output, dim=1).item()
        
        # Zero gradients everywhere
        self.model.zero_grad()
        
        # Set the output for the target category to 1, and 0 for other categories
        one_hot_output = torch.zeros((1, model_output.shape[-1]))
        one_hot_output[0][target_category] = 1
        
        # Backward pass to get gradient information
        model_output.backward(gradient=one_hot_output)
        
        # Get the target layer's output after the forward pass
        target_layer_output = self.outputs[0]
        
        # Global Average Pooling (GAP) to get the weights
        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        # Weighted combination to get the attention map
        cam = torch.zeros(target_layer_output.shape[1:]).to(input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * target_layer_output[i, :, :]
        
        # ReLU to get only the positive values
        cam = nn.ReLU()(cam)
        
        # Resize the CAM to the input tensor's size
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode="bilinear").squeeze().cpu().detach().numpy()
        
        return cam