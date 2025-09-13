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

from Task2_YOLO_RESNET50_LSTM_LOADERS import Task2_Loader
from Task2_YOLO_RESNET50_LSTM import Model


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    video_path = "/mounts/tf2/Task2_docker/Inputs/" 
    output_path = "/mounts/tf2/Task2_docker/Outputs/"
    video_files = [os.path.splitext(item)[0] for item in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, item)) and item.endswith('.mp4')]
    video_column = 'VIDEO'

    df = pd.DataFrame(video_files, columns=[video_column])

    video_loader = Task2_Loader(device, df, video_path).get_loader()

    NUM_OSATS_CATEGORIES = 8
    NUM_SCORES_PER_CATEGORY = 5

    model_params = {
        'detector_score_thresh' : 0.75,     
        'target_yolo_class_idx' :None,
        'num_frames_for_lstm' : 128, 
        'lstm_hidden_size' : 2024,
        'lstm_num_layers':3,
        'lstm_dropout':0.6
    }

    model = Model(device, video_loader, NUM_OSATS_CATEGORIES,NUM_SCORES_PER_CATEGORY,"./Task2_best_model.pth", **model_params)
    predictions = model.predict()

    data_for_df = []
    for video_name, pred_scores in zip(video_files, predictions):
        row = [video_name] + list(pred_scores)
        data_for_df.append(row)

    column_names = ['VIDEO', 'OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT',
    'OSATS_SUTURE','OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY']
    
    results_df = pd.DataFrame(data_for_df, columns=column_names)
    video_files_np = np.array(video_files).reshape(-1, 1) 

    combined_data = np.hstack((video_files_np, predictions))

    results_df = pd.DataFrame(combined_data, columns=column_names)
   
    results_df.to_csv(os.path.join(output_path, 'predictions.csv'), index=False, sep=';')
    
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
