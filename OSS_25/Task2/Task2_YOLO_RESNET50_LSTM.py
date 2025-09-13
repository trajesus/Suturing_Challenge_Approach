import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

# Imports de Machine Learning e Visão Computacional
import av
import kornia
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize

resnet_roi_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_best_roi_from_yolo_pred_and_crop(
    yolo_pred_tensor,
    original_frame_chw_on_device: torch.Tensor,
    score_threshold: float = 0.75,
    target_class_idx: int = None,
    min_roi_size: int = 10
) -> torch.Tensor:
    best_box_coords = None
    max_score = score_threshold
    if yolo_pred_tensor is not None and len(yolo_pred_tensor) > 0:
        for det in yolo_pred_tensor:
            if det[4].item() > max_score and (target_class_idx is None or int(det[5].item()) == target_class_idx):
                max_score = det[4].item()
                best_box_coords = tuple(det[0:4].cpu().numpy())
    C, H, W = original_frame_chw_on_device.shape
    if best_box_coords:
        xmin, ymin, xmax, ymax = best_box_coords
        top, left = int(round(ymin)), int(round(xmin))
        height, width = int(round(ymax - ymin)), int(round(xmax - xmin))
        if height >= min_roi_size and width >= min_roi_size:
            return TF.crop(original_frame_chw_on_device, top, left, height, width)
    ph, pw = H // 2, W // 2
    pt, pl = (H - ph) // 2, (W - pw) // 2
    return TF.crop(original_frame_chw_on_device, pt, pl, ph, pw)


class Score_Classifier_Heads_Task2(nn.Module):
    def __init__(self, num_osats_categories, num_scores_per_osats_cat, input_features_dim):
        super().__init__()
        self.osats_heads = nn.ModuleList([
            nn.Linear(input_features_dim, num_scores_per_osats_cat)
            for _ in range(num_osats_categories)
        ])
    def forward(self, features_batch):
        head_outputs = [head(features_batch) for head in self.osats_heads]
        return torch.stack(head_outputs, dim=2) # Shape: B, Num_Scores, Num_Categories

class TwoStage_VideoScoringModel_YOLO_ResNet_LSTM_Task2(nn.Module):
    def __init__(self, device, num_osats_categories, num_scores_per_category, yolo_model_path='yolo_preTrained.pt', **kwargs):
        super().__init__()
        self.device = device

        self.register_buffer('mean_tensor', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std_tensor', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.detector_score_threshold = kwargs.get('detector_score_thresh', 0.75)
        self.target_yolo_class_idx = kwargs.get('target_yolo_class_idx', None)
        self.num_frames_for_lstm = kwargs.get('num_frames_for_lstm', 5)
        lstm_hidden_size = kwargs.get('lstm_hidden_size', 512)
        lstm_num_layers = kwargs.get('lstm_num_layers', 4)
        lstm_dropout = kwargs.get('lstm_dropout', 0.6)

        try:
            yolo_local_repo_path = './yolov5' 
            self.detector = torch.hub.load(yolo_local_repo_path, 'custom', path=yolo_model_path,source='local').to(device)
            self.detector.eval()
            for param in self.detector.parameters(): param.requires_grad = False
        except Exception as e: self.detector = None; print(f"Erro ao carregar YOLO: {e}")

        #self.roi_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.roi_feature_extractor = models.resnet50(weights=None) 

        resnet_local_weights_path = 'resnet50-weights.pth'

        if os.path.exists(resnet_local_weights_path):
            print("Carregar pesos ResNet 50 de: {resnet_local_weights_path}")
            self.roi_feature_extractor.load_state_dict(torch.load(resnet_local_weights_path))
        else:
            print(f"AVISO: Ficheiro de pesos '{resnet_local_weights_path}' não encontrado. A ResNet não será pré-treinada.")
        
        self.backbone_out_features = self.roi_feature_extractor.fc.in_features
        self.roi_feature_extractor.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.backbone_out_features, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        self.score_classifier = Score_Classifier_Heads_Task2(
            num_osats_categories, num_scores_per_category, lstm_hidden_size
        )

        self.roi_transform = resnet_roi_transform

    def denormalize_frame_for_detector(self, normalized_batch):
        return (normalized_batch * self.std_tensor) + self.mean_tensor
    
    def forward(self, video_clips_batch):
        B, T, C, H_proc, W_proc = video_clips_batch.shape

        if C == 4:
            video_clips_batch = video_clips_batch[:, :, :3, :, :]
            C = 3
        
        actual_num_frames_to_sample = min(T, self.num_frames_for_lstm)
        
        if actual_num_frames_to_sample <= 0:
            dummy_lstm_input = torch.zeros(B, self.num_frames_for_lstm, self.backbone_out_features, device=self.device)
            _, (h_n, _) = self.lstm(dummy_lstm_input)
            return self.score_classifier(h_n[-1])
        
        frame_indices = torch.linspace(0, T - 1, steps=actual_num_frames_to_sample, device=self.device).long()
        selected_frames = video_clips_batch[:, frame_indices]
        individual_frames = selected_frames.contiguous().view(B * actual_num_frames_to_sample, C, H_proc, W_proc)

        frames_for_yolo = self.denormalize_frame_for_detector(individual_frames)

        with torch.no_grad():
            yolo_preds_tensor_list = self.detector(frames_for_yolo, size=W_proc)
        
        processed_rois = [self.roi_transform(get_best_roi_from_yolo_pred_and_crop(p, f)) for p, f in zip(yolo_preds_tensor_list, frames_for_yolo)]

        if not processed_rois:
            dummy_lstm_input = torch.zeros(B, self.num_frames_for_lstm, self.backbone_out_features, device=self.device)
            _, (h_n, _) = self.lstm(dummy_lstm_input)
            return self.score_classifier(h_n[-1])
        roi_features = self.roi_feature_extractor(torch.stack(processed_rois))

        sequence_features = roi_features.view(B, actual_num_frames_to_sample, -1)
        if actual_num_frames_to_sample < self.num_frames_for_lstm:
            padding = torch.zeros(B, self.num_frames_for_lstm - actual_num_frames_to_sample, self.backbone_out_features, device=self.device)
            sequence_features = torch.cat([sequence_features, padding], dim=1)
        _, (h_n, _) = self.lstm(sequence_features)
        return self.score_classifier(h_n[-1])


class Model:
    def __init__(self,device, loader, num_osats_categories, num_scores_per_category, model_path="./Task2_best_model.pth", **kwargs):
        self.set_seeds(seed=0)
        self._device = device
        self._video_loader = loader
        self._num_osats_categories = num_osats_categories
        self._num_scores_per_category = num_scores_per_category
        
        self._model = TwoStage_VideoScoringModel_YOLO_ResNet_LSTM_Task2(
            device=self._device,
            num_osats_categories=num_osats_categories,
            num_scores_per_category=num_scores_per_category,
            **kwargs
        )

        if model_path and os.path.exists(model_path):
            print(f"A carregar state_dict de: {model_path}")
    
            checkpoint_state_dict = torch.load(model_path, map_location=self._device)
            
            cleaned_state_dict = {}
            
            for key, value in checkpoint_state_dict.items():
                if not key.startswith('detector.'):
                    cleaned_state_dict[key] = value
                    
            self._model.load_state_dict(cleaned_state_dict, strict=False)
        
        
        self._model = self._model.to(self._device)
        
    def set_seeds(self, seed=0):
        np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    def evaluate_model(self): 
        self._model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_predictions, all_true_labels = [], []
        
        # --- Passo 1: Recolher todas as predições e labels ---
        with torch.no_grad():
            for videos, labels in tqdm(self._video_loader, desc="Avaliando"):
                videos, labels = videos.to(self._device), labels.to(self._device, dtype=torch.long)
                
                outputs = self._model(videos) # Shape: (B, Num_Scores, Num_Categories)
                
                # Calcula a perda média para o lote
                batch_loss = sum(criterion(outputs[:, :, i], labels[:, i]) for i in range(self._num_osats_categories))
                total_loss += (batch_loss / self._num_osats_categories).item() * videos.size(0)
                
                # Obtém as predições (índice da maior pontuação)
                preds = torch.argmax(outputs, dim=1) # Shape: (B, Num_Categories)
                
                all_predictions.append(preds.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())

        # --- Passo 2: Converter listas para arrays NumPy para cálculo de métricas ---
        # Ambos terão shape: (Num_Amostras, Num_Categorias)
        predictions_np = np.concatenate(all_predictions)
        true_labels_np = np.concatenate(all_true_labels)
        
        if len(self._video_loader.dataset) == 0:
            print("Dataset de avaliação vazio. Não é possível calcular métricas.")
            return 0.0

        avg_loss = total_loss / len(self._video_loader.dataset)

        # --- Passo 3: Calcular e imprimir as métricas detalhadas ---
        
        # Nomes das categorias para relatórios mais legíveis
        osats_category_names = [
            'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE',
            'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATSFINALQUALITY'
        ]

        per_category_accuracy = []
        per_category_mae = []
        
        print("\n--- Relatório de Classificação Detalhado por Categoria OSATS ---")
        for i in range(self._num_osats_categories):
            cat_name = osats_category_names[i] if i < len(osats_category_names) else f"Categoria_{i+1}"
            true_cat_labels = true_labels_np[:, i]
            pred_cat_labels = predictions_np[:, i]
            
            # Calcula métricas para esta categoria
            accuracy = accuracy_score(true_cat_labels, pred_cat_labels)
            mae = mean_absolute_error(true_cat_labels, pred_cat_labels)
            per_category_accuracy.append(accuracy)
            per_category_mae.append(mae)
            
            # Imprime o relatório de classificação
            print(f"\n-- {cat_name} (Acurácia: {accuracy:.4f}, MAE: {mae:.4f}) --")
            report = classification_report(
                true_cat_labels, 
                pred_cat_labels, 
                labels=range(self._num_scores_per_category),
                target_names=[f"Pontuação {j}" for j in range(self._num_scores_per_category)],
                zero_division=0,
                digits=3
            )
            print(report)

        avg_accuracy_across_categories = np.mean(per_category_accuracy) if per_category_accuracy else 0.0
        avg_mae_across_categories = np.mean(per_category_mae) if per_category_mae else 0.0

        num_exact_matches = np.sum(np.all(true_labels_np == predictions_np, axis=1))
        exact_match_ratio = num_exact_matches / true_labels_np.shape[0]

        flat_predictions = predictions_np.flatten()
        flat_true_labels = true_labels_np.flatten()
        overall_f1_weighted = f1_score(flat_true_labels, flat_predictions, average='weighted', zero_division=0)
        
        print("\n--- Métricas de Avaliação Agregadas ---")
        print(f"Loss Médio (CrossEntropy): {avg_loss:.4f}")
        print(f"Mean Absolute Error (MAE Médio por Categoria): {avg_mae_across_categories:.4f}")
        print(f"F1 Score (Agregado Ponderado): {overall_f1_weighted:.4f}")
        print(f"Acurácia (Média por Categoria OSATS): {avg_accuracy_across_categories:.4f}")
        print(f"Acurácia (Exact Match Ratio - Todas as 8 categorias corretas): {exact_match_ratio:.4f}")

        return avg_loss, avg_mae_across_categories, overall_f1_weighted, avg_accuracy_across_categories

    def predict(self): 
        self._model.eval(); all_predictions = []
        with torch.no_grad():
            for batch in tqdm(self._video_loader, desc="Predizendo"):
                videos = (batch[0] if isinstance(batch, (list, tuple)) else batch)
                videos = videos.to(self._device)
                
                outputs = self._model(videos)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.append(preds.cpu().numpy())
        return np.concatenate(all_predictions)

