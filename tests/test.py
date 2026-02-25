import torch

#print torch version
print(torch.__version__)
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. CPU will be used.")