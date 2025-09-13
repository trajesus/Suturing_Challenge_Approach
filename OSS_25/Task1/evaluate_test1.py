import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import seaborn as sns

import av
import json 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import Normalize, Compose, ToTensor
from torchvision import transforms
import decord
import random
import kornia

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import time
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Task1_InceptionV3_FinalModel import Model
from Task1_InceptionV3_Loaders import Task1_Loader

"""
def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    video_path = "/mounts/OSS_dataset/Train_Task1_2/"
    labels_path = "/mounts/OSS_dataset/Dataset_2024/Train/OSATS.xlsx"
    df = pd.read_excel(labels_path)
    df = df.head(50)

    video_files = [os.path.splitext(f)[0] for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and f.endswith(('.mp4', '.avi', '.mov'))]
    video_column = 'VIDEO'

    df = df[df[video_column].isin(video_files)]
    
    video_loader = Task1_Loader(device, df, video_path).get_loader()

    num_classes = 4
    model = Model(device, video_loader, num_classes)
    loss, mae, f1, accuracy = model.evaluate_model()
    
    print(f"Evaluation Results - Loss: {loss}, MAE: {mae}, F1 Score: {f1}, Accuracy: {accuracy}")
"""

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # video_path = "/mounts/tf2/Task1_docker/Inputs/"
    video_path = "/mounts/OSS_dataset/Train_Task1_2/"
    labels_path = "/mounts/OSS_dataset/Dataset_2024/Train/OSATS.xlsx"
    df = pd.read_excel(labels_path)
    df = df.head(10)

    video_files = [os.path.splitext(f)[0] for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and f.endswith(('.mp4', '.avi', '.mov'))]
    video_column = 'VIDEO'

    df = df[df[video_column].isin(video_files)]
    
    video_loader = Task1_Loader(device, df, video_path).get_loader()

    num_classes = 4
    model = Model(device, video_loader, num_classes)
    loss, mae, f1, accuracy = model.evaluate_model()
    
    print(f"Evaluation Results - Loss: {loss}, MAE: {mae}, F1 Score: {f1}, Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
