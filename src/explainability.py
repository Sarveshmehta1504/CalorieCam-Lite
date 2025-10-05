import torch, cv2, numpy as np
from torchvision import transforms

# Simple Grad-CAM for ResNet last conv layer
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # hook last conv layer in resnet backbone[-1] is AdaptiveAvgPool2d -> not good
        # resnet children: layer1, layer2, layer3, layer4, avgpool
        self.target_layer = self.model.embed.backbone[-2]  # layer4
        self.handler_f = self.target_layer.register_forward_hook(self.forward_hook)
        self.handler_b = self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx, orig_img):
        self.model.zero_grad()
        logits, _ = self.model(input_tensor)
        score = logits[:, class_idx].squeeze()
        score.backward(retain_graph=True)

        grads = self.gradients  # [B,C,H,W]
        acts = self.activations # [B,C,H,W]
        weights = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
        heatmap = (cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET) * 0.4 + orig_img * 0.6).astype(np.uint8)
        return heatmap

    def close(self):
        self.handler_f.remove()
        self.handler_b.remove()
