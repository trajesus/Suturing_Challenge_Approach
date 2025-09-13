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


class VideoDataset(Dataset):
    def __init__(self, df_dataset, cfg, dataset_path, transformer, mean, std) -> None:
        super().__init__()
        self.df_dataset = df_dataset.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.cfg = cfg
        self.label = "GLOBA_RATING_SCORE"
        self.mean = mean
        self.std = std
        self.transformer = transformer

    def __len__(self):
        return len(self.df_dataset)

    def set_task_one(self, has_label=True):
        if has_label:
            self.label = "GLOBA_RATING_SCORE"
        else:
            self.label = None

    def set_task_two(self, has_label=True):
        if has_label:
            self.label = [
                'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT',
                'OSATS_SUTURE', 'OSATS_FLOW', 'OSATS_KNOWLEDGE',
                'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY'
            ]
        else:
            self.label = None
    
    def to_show(self, frame_tensor, show_filter=False):
        """
        Receives a single processed frame tensor from this dataset and displays it.
        Just a helper function to visualize the frames.
        
        Args:
            frame_tensor (Tensor): A single frame tensor of shape [4, H, W].
            show_filter (bool): If True, displays the edge filter channel. 
                                If False, displays the original RGB image.
        """
        if not isinstance(frame_tensor, torch.Tensor) or frame_tensor.ndim != 3 or frame_tensor.shape[0] != 4:
            raise ValueError("Input must be a 4-channel tensor [4, H, W] from this dataset.")
   
        mean = self.mean.view(4, 1, 1)
        std = self.std.view(4, 1, 1)
        
        denorm_frame = frame_tensor.cpu() * std + mean
        
        # Clamp values to the valid [0, 1] range for display
        denorm_frame = torch.clamp(denorm_frame, 0, 1)
        
        if show_filter:
            filter_channel = denorm_frame[3, :, :] # Shape: [H, W]
           
            image_to_display = filter_channel.numpy()
            cmap = 'gray'
            title = 'Edge Filter Channel'
        else:
            
            rgb_channels = denorm_frame[:3, :, :] # Shape: [3, H, W]
           
            image_to_display = rgb_channels.permute(1, 2, 0).numpy()
            cmap = None 
            title = 'Original RGB Image'
      
        plt.imshow(image_to_display, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def __getitem__(self, idx):
        video_info = self.df_dataset.iloc[idx]
        video_name = video_info["VIDEO"] + ".mp4"
        video_path = os.path.join(self.dataset_path, video_name)
        
        target_T = self.cfg["model_input_frames"]
    
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames
        except Exception as e:
            print(f"Error opening or reading video: {video_path}. Reason: {e}")
            print("Returning an empty tensor for this item.")
          
            C, H, W = 4, self.cfg["target_frame_height"], self.cfg["target_frame_width"]
            empty_frames = torch.zeros((target_T, C, H, W))
            dummy_label = torch.zeros(len(self.label)) if isinstance(self.label, list) else torch.tensor(0.0)
            return empty_frames, dummy_label
     
        if total_frames == 0:
            print(f"Warning: Video {video_name} has 0 frames. Returning an empty tensor.")
            container.close()
            C, H, W = 4, self.cfg["target_frame_height"], self.cfg["target_frame_width"]
            empty_frames = torch.zeros((target_T, C, H, W))
            dummy_label = torch.zeros(len(self.label)) if isinstance(self.label, list) else torch.tensor(0.0)
            return empty_frames, dummy_label
      
        if total_frames < target_T:
            print(f"Warning: Video {video_name} has only {total_frames} frames, "
                  f"which is less than the required {target_T}. Frames will be repeated.")
      
        sampling_power = self.cfg.get("frame_sampling_power", 1.0)
    
        linear_space = np.linspace(0, 1, target_T)
        
        nonlinear_space = 1 - (1 - linear_space) ** sampling_power
        
        indices = np.round(nonlinear_space * (total_frames - 1)).astype(int)

        frames_list = []
      
        extracted_frames_dict = {}
        indices_to_get = set(indices) 
    
        frame_idx_counter = 0
        for frame in container.decode(video=0):
            if frame_idx_counter in indices_to_get:
                pil_image = frame.to_image().convert('RGB')
                transformed_frame = self.transformer(pil_image)
                extracted_frames_dict[frame_idx_counter] = transformed_frame
                
                indices_to_get.remove(frame_idx_counter)
                if not indices_to_get:
                    break 
            
            frame_idx_counter += 1
        
        container.close()
        
        for index_val in indices:
            if index_val in extracted_frames_dict:
                frames_list.append(extracted_frames_dict[index_val])
            else:
                print(f"Error: Frame index {index_val} was requested but not found in video {video_name}. "
                      f"This might indicate a video decoding issue. Appending a zero tensor.")
                C, H, W = 4, self.cfg["target_frame_height"], self.cfg["target_frame_width"]
                frames_list.append(torch.zeros((C, H, W)))
    
        if not frames_list:
            print(f"Error: No frames were extracted for video {video_name}. Returning an empty tensor.")
            C, H, W = 4, self.cfg["target_frame_height"], self.cfg["target_frame_width"]
            frames = torch.zeros((target_T, C, H, W))
        else:
            frames = torch.stack(frames_list, dim=0)

        if self.label is not None:
            if isinstance(self.label, str):
                label = video_info[self.label]
                label = torch.tensor(label, dtype=torch.long) 
            else:
                label = [video_info[col] for col in self.label]
                label = torch.tensor(label, dtype=torch.float32)
    
            return frames, label
    
        return frames
    
    
class AddKorniaEdgeChannel:
    """
    A custom transform that uses Kornia on the GPU to calculate the Sobel
    edge map and append it as a fourth channel.
    """
    def __init__(self, device):
        """
        Initializes the transform.
        Args:
            device (torch.device): The device (e.g., 'cuda:0') to perform operations on.
        """
        self.device = device

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (Tensor): A CPU tensor of size (3, H, W).
        Returns:
            Tensor: A CPU tensor with the edge channel, size (4, H, W).
        """
        if img_tensor.shape[0] != 3:
            raise ValueError("Input tensor must have 3 channels (RGB).")


        # 2. Perform all subsequent operations on the GPU
        img_tensor_batch = img_tensor.unsqueeze(0)
        gray_tensor_batch = kornia.color.rgb_to_grayscale(img_tensor_batch)
        edge_batch = kornia.filters.sobel(gray_tensor_batch)
        
        edge_magnitude = edge_batch.squeeze(0)

        # Normalization also happens on the GPU
        min_val, max_val = edge_magnitude.min(), edge_magnitude.max()
        edge_normalized = (edge_magnitude - min_val) / (max_val - min_val + 1e-6)

        # Concatenate the original GPU tensor and the new edge tensor
        output_tensor = torch.cat([img_tensor, edge_normalized], dim=0)
        
        # 3. CRITICAL: Move the final result back to the CPU before returning
        # The DataLoader worker needs a CPU tensor to collate with other samples.
        return output_tensor.cpu()


class Task1_Loader:
    def __init__(self, device, df, video_path, batch_size=2, size=(299, 299), num_workers=4,
                 mean=torch.tensor([0.485, 0.456, 0.406, 0.5]), std=torch.tensor([0.229, 0.224, 0.225, 0.5])):
        self._video_path = video_path
        self._df = df
        self._df_has_labels = df.columns.str.contains('GLOBA_RATING_SCORE').any()
        self._size = size
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._mean = mean
        self._std = std

        self._cfg = {
            "sampling_strategy": "fixed_step_sampling",
            "frame_division_factor": 75,
            "model_input_frames": 128,
            "target_frame_height": self._size[0],
            "target_frame_width": self._size[1],
            "frame_sampling_power": 2.5
        }
        
        self._transformer = Compose([
            ToTensor(),
            transforms.Resize(self._size, antialias=True),  
            AddKorniaEdgeChannel(device=device),
            Normalize(mean=self._mean, std=self._std)
        ])

        if self._df_has_labels:
            self._df = self._update_df_with_binned_grs(self._df)

    def _set_seeds(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def _get_global_rating_score(self, video: str):
        try:
            score = self._df[self._df['VIDEO'] == video]['GLOBA_RATING_SCORE'].iloc[0]
            if score < 16:
                score = 0
            elif score < 24:
                score = 1
            elif score < 32:
                score = 2
            elif score < 40:
                score = 3
            return score
        except:
            print(f"Video {video} not found")
            return None 

    def _get_vector_of_ratings(self, video: str):
        video_data = self._df[self._df['VIDEO'] == video]
        osats_columns = ['OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT',
                        'OSATS_SUTURE', 'OSATS_FLOW', 'OSATS_KNOWLEDGE',
                        'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY']
        return video_data[osats_columns].iloc[0].values.tolist()

    def _update_df_with_binned_grs(self, df):
        updated_df = df.copy()
        binned_scores = updated_df['VIDEO'].apply(self._get_global_rating_score)
        updated_df['GLOBA_RATING_SCORE'] = binned_scores
        
        if binned_scores.isnull().any():
            print("Warning: Some videos were not found. Corresponding GRS values are None and will be dropped.")
            updated_df = updated_df.dropna(subset=['GLOBA_RATING_SCORE']) 
        
        return updated_df
    
    def _prepare_data_loaders(self, task=1): 
        dataset = VideoDataset(self._df, self._cfg, self._video_path, self._transformer, self._mean, self._std)
        
        dataset.set_task_one(has_label=('GLOBA_RATING_SCORE' in self._df.columns)) if task == 1 else dataset.set_task_two(has_label=(['OSATS_RESPECT', 'OSATS_MOTION', 
                                                                                                                                      'OSATS_INSTRUMENT', 'OSATS_SUTURE', 
                                                                                                                                      'OSATS_FLOW', 'OSATS_KNOWLEDGE', 
                                                                                                                                      'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY'] 
                                                                                                                                     in self._df.columns))

        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True
        )

        return loader
    
    def get_loader(self):
        return self._prepare_data_loaders(task=1)
