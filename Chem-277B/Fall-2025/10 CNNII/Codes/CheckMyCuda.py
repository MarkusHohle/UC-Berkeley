# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:01:51 2024

@author: MMH_user
"""

#if cuda is not recognized, run:
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


import torch

def test_cuda():
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    print("CUDA Available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs: ", torch.cuda.device_count())
        print("GPU Name: ", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    test_cuda()