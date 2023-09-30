import pickle
from typing import Any
import numpy as np
import pandas as pd
import random
from skimage import io
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from glob import glob

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

import torch.nn as nn


IM_SIZE = 224
MODEL_VERSION = 'dinov2_vits14'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoVisionTransformerClassifier(nn.Module):
    '''Neural network'''
    def __init__(self):
        super().__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', MODEL_VERSION)
        self.quality_classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.quality_classifier(x)
        return x


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        try:
            img = Image.open(f).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        return img

class Model():
    '''API class for neural model'''
    def __init__(self, weights_path) -> None:
        self.loader = pil_loader

        self.transform = transforms.Compose([
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = DinoVisionTransformerClassifier().to(DEVICE)
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.model.eval()

    def __call__(self, files, batch_size=16) -> Any:
        outputs = []
        with torch.no_grad():
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i + batch_size]

                batch_inputs = [self.transform(self.loader(f)) for f in batch_files]
                batch_inputs = torch.stack(batch_inputs).to(DEVICE)

                probs = torch.softmax(self.model(batch_inputs), 1).detach().cpu().numpy()
                output = np.zeros_like(probs)
                output[np.arange(output.shape[0]), np.argmax(probs, 1)] = 1

                outputs.append(output)
        return np.vstack(outputs)