import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Should return True if CUDA is working
print(torch.cuda.device_count())  # Should be >0 if GPU is detected
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")  # GPU name
