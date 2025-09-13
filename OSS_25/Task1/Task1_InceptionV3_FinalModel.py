#!pip install decord openpyxl av

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


decord.bridge.set_bridge('torch')


class SutureQualityNetInceptionV3(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        """
        Inicializa o modelo com a arquitetura InceptionV3.
        """
        super(SutureQualityNetInceptionV3, self).__init__()
        
        # 1. Carrega o modelo InceptionV3 pré-treinado
        # self.feature_extractor = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        # torch.save(self.feature_extractor.state_dict(), './inception_v3_pretrained.pth')
        
        self.feature_extractor = models.inception_v3(weights=None, aux_logits=True)
        self.feature_extractor.load_state_dict(torch.load('./inception_v3_pretrained.pth'))

        self.feature_extractor.transform_input = False
        print("Transformação de entrada interna do InceptionV3 desativada.")

        # 2. Adapta a primeira camada convolucional para aceitar 4 canais (RGB + Edge)
        original_conv1 = self.feature_extractor.Conv2d_1a_3x3.conv
        original_weights = original_conv1.weight.data
        
        new_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_weights
            new_conv1.weight[:, 3, :, :] = original_weights.mean(dim=1)
            
        self.feature_extractor.Conv2d_1a_3x3.conv = new_conv1
        print("Sucesso ao adaptar a primeira camada da InceptionV3 para 4 canais (RGB + Edge).")
        
        # 3. Substitui a camada final de classificação (fully connected)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        print(f"Classificador final substituído por um novo com Dropout (taxa={dropout_rate}) e {num_classes} classes.")
        
        # O classificador auxiliar também precisa ser ajustado
        if self.feature_extractor.AuxLogits is not None:
            num_ftrs_aux = self.feature_extractor.AuxLogits.fc.in_features
            self.feature_extractor.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
            print("Classificador auxiliar também foi ajustado.")


    def forward(self, x):
        """
        Executa a passagem para a frente (forward pass).
        """
        if x.shape[1] != 4:
            raise ValueError(f"Input de imagem esperado com 4 canais, mas recebeu {x.shape[1]}.")
        
        out = self.feature_extractor(x)
        return out


class Model:
    def __init__(self, device, loader, num_classes, model_path="./SutureQuality_InceptionV3_best_model.pth"):
        self.set_seeds(seed=0)
        self._device = device
        self._video_loader = loader
        self._num_classes = num_classes
        
        self._model = SutureQualityNetInceptionV3(num_classes=self._num_classes, dropout_rate=0.5)

        try:
            self._model.load_state_dict(torch.load(model_path)) 
        except FileNotFoundError:
            print(f"Error: The models' state_dict could not be found in {model_path}.")
            exit()
            
        self._model = self._model.to(self._device)
        
    def set_seeds(self, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)  

    def evaluate_model(self): 
        self._model.eval()
    
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for videos, labels in tqdm(self._video_loader, desc="Evaluating"):
                videos, labels = videos.to(self._device), labels.to(self._device, dtype=torch.long)
                
                batch_size_actual, num_frames_actual, C, H, W = videos.shape
                videos_reshaped = videos.view(batch_size_actual * num_frames_actual, C, H, W)
                
                outputs_frames = self._model(videos_reshaped)
                outputs = outputs_frames.view(batch_size_actual, num_frames_actual, -1).mean(dim=1)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item() * batch_size_actual

                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        test_loss /= len(self._video_loader.dataset)
        test_mae = mean_absolute_error(true_labels, predictions)
        test_f1 = f1_score(true_labels, predictions, average='weighted')
        test_accuracy = accuracy_score(true_labels, predictions)

        print(f"Test Loss (CrossEntropy): {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Test F1 Score (weighted): {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")

        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(self._num_classes), yticklabels=range(self._num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, zero_division=0, labels=[i for i in range(self._num_classes)], target_names=[f'Class {i}' for i in range(self._num_classes)]))

        return test_loss, test_mae, test_f1, test_accuracy
    
    def predict(self): 
        self._model.eval()

        predictions = []

        with torch.no_grad():
            for videos in tqdm(self._video_loader, desc="Predicting"):
                videos = videos.to(self._device)
                
                batch_size_actual, num_frames_actual, C, H, W = videos.shape
                videos_reshaped = videos.view(batch_size_actual * num_frames_actual, C, H, W)
                
                outputs_frames = self._model(videos_reshaped)
                outputs = outputs_frames.view(batch_size_actual, num_frames_actual, -1).mean(dim=1)
                                                                    
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions
