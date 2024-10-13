import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

# Load the ONNX model
ort_session = ort.InferenceSession('guitarid_model_v0.0.1.onnx')

# Load the same image for comparison
img = Image.open('guitar_dataset/val/Ibanez RG370DX/59.jpg')

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = preprocess(img).unsqueeze(0)

print(img_tensor)

# Convert PyTorch tensor to NumPy
img_numpy = img_tensor.numpy()

# Run ONNX inference
ort_inputs = {ort_session.get_inputs()[0].name: img_numpy}
ort_outs = ort_session.run(None, ort_inputs)

# # Get the PyTorch prediction
# model.eval()
# with torch.no_grad():
#     pytorch_out = model(img_tensor)

print(f'ONNX output: {ort_outs}')
# print(f'PyTorch output: {pytorch_out}')
