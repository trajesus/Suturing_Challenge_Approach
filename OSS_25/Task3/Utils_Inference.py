# -*- coding: utf-8 -*-

'''
File for inference helper funcs
'''
#imports
import csv
import cv2
import joblib
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import re
import SimpleITK as sitk
import sys
import time

from collections import defaultdict
import os
from os import listdir
from os.path import exists, join
from PIL import Image
from radiomics import featureextractor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional, Tuple
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor

##### Radiomics and models paths setup
BASE_PATH_RADIOMICS = "./radiomics" # local path
BASE_PATH_CHECKPOINTS = "./checkpoints" # local path
EXTRACTOR_PARAMS_FILE = f"{BASE_PATH_RADIOMICS}/Parameters.yml"

# Mapping each tool to its radiomics CSV
tool_paths = {
    'left_hand': f'{BASE_PATH_RADIOMICS}/_left_hand_segment_mask.png_radiomics.csv',
    'right_hand': f'{BASE_PATH_RADIOMICS}/_right_hand_segment_mask.png_radiomics.csv',
    'tweezers': f'{BASE_PATH_RADIOMICS}/_tweezers_mask.png_radiomics.csv',
    'scissors': f'{BASE_PATH_RADIOMICS}/_scissors_mask.png_radiomics.csv',
    'needle_holder': f'{BASE_PATH_RADIOMICS}/_needle_holder_mask.png_radiomics.csv',
    'needle': f'{BASE_PATH_RADIOMICS}/_needle_mask.png_radiomics.csv',
    'needle1': f'{BASE_PATH_RADIOMICS}/_needle1_mask.png_radiomics.csv',
    'others': f'{BASE_PATH_RADIOMICS}/others_radiomics.csv',
    }

tool_label_map = {
    'left_hand': 0,
    'right_hand': 0,
    'tweezers': 1,
    'scissors': 2,
    'needle_holder': 3,
    'needle': 4,
    'needle1': 4,
    'others' : 5
    }

target_names = [
    'hand',
    'tweezers',
    'scissors',
    'needle_holder',
    'needle',
    'others'
    ]

##### SAM2
sam2_checkpoint = f"{BASE_PATH_CHECKPOINTS}/SAM2/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

#### Data preparation functions
def prepare_dataset(feature_data: list, case_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if not feature_data:
        return pd.DataFrame(), pd.Series(dtype=object)
    rows = []
    for i, feat in enumerate(feature_data):
        numeric = {}
        for key, val in feat.items():
            if key.startswith("diagnostics_"):
                continue
            # try numeric conversion
            try:
                # handle numpy types & booleans
                if isinstance(val, (np.floating, float, np.integer, int)):
                    numeric[key] = float(val)
                elif isinstance(val, (np.ndarray, list)):
                    # rare: multi-value feature -> take mean as fallback
                    numeric[key] = float(np.mean(val))
                elif isinstance(val, str):
                    # some features may be strings (shouldn't happen for non-diagnostics) -> try cast
                    numeric[key] = float(val)
                else:
                    numeric[key] = float(val)
            except Exception:
                # fallback: keep zero and warn
                numeric[key] = 0.0
                print(f"Warning: non-numeric feature {key} (type {type(val)}). Setting 0.0")
        rows.append(numeric)
    df = pd.DataFrame(rows).fillna(0.0)
    # keep features with prefixes original, log, wavelet (case-insensitive)
    feature_cols = [c for c in df.columns if any(p in c.lower() for p in ['original', 'log', 'wavelet'])]
    case_names = pd.Series([f"{case_name}_mask{i+1}" for i in range(len(feature_data))])
    return df[feature_cols], case_names

class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, case_names: pd.Series):
        self.df = df.reset_index(drop=True)
        self.feature_names = [col for col in self.df.columns if 'original' in col.lower()]  # adjust as needed
        self.X = self.df[self.feature_names].values.astype('float32') if self.feature_names else np.zeros((len(self.df),0),dtype=np.float32)
        self.case_names = case_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.case_names.iloc[idx]

#### UNET
VERBOSE = True
UNET_MODEL = True

### **Define the dicts to map the values and the functions to load and view the data**
# Define class and keypoint mappings based on dataset description
CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Scissors",
    3: "Tweezers",
    4: "Needle_Holder",
    5: "Needle"
    }

HAND_KEYPOINTS = {
    0: "Thumb",
    1: "Middle",
    2: "Index",
    3: "Ring",
    4: "Pinky",
    5: "Back of hand"
    }

TOOL_KEYPOINTS = {
    "Scissors": {0: "Left (Sharp point)", 1: "Right (Broad point)", 2: "Joint"},
    "Tweezers": {0: "Left (with Nub)", 1: "Right (with Hole)", 2: "Nub"},
    "Needle_Holder": {0: "Left", 1: "Right (Right when text visible)", 2: "Joint"},
    "Needle": {0: "Left (End)", 1: "Right (Tip)", 2: "Middle"}
    }

mask_naming = {
    "Left Hand": ["left_hand_segment_mask"],
    "Right Hand": ["right_hand_segment_mask"],
    "Scissors": ["scissors_mask"],
    "Tweezers": ["tweezers_mask"],
    "Needle_Holder": ["needle_holder_mask"],
    "Needle":["needle_mask", "needle1_mask"]
    }

class_id = {
    "Left Hand": 0,
    "Right Hand": 1,
    "Scissors": 2,
    "Tweezers": 3,
    "Needle_Holder": 4,
    "Needle": 5
    }

num_keypoints_dict = {
    "Left Hand": 6,
    "Right Hand": 6,
    "Scissors": 3,
    "Tweezers": 3,
    "Needle_Holder": 3,
    "Needle": 3
    }

# UNet model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, features=[64, 128, 256, 512], dropout=0.0):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_rate = dropout
        
        # Encoder
        for feature in features:
            self.encoder_layers.append(self._conv_block(in_channels, feature, dropout))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1]*2, dropout)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        reversed_features = features[::-1]
        prev_channels = features[-1]*2
        for feature in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2))
            self.decoder_layers.append(self._conv_block(prev_channels, feature, dropout))
            prev_channels = feature  # next iteration input channels

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):
        skip_connections = []
        # Encoder
        for enc in self.encoder_layers:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_layers[idx](x)
        
        heatmaps = self.final_conv(x)                        # (B, K, H, W)
        
        return heatmaps#, vis_logits

def generator(case_np, mask_np, shape_pad):
        # Remove extra singleton dimension if present
        if case_np.ndim == 4 and case_np.shape[-1] == 1:
            case_np = np.squeeze(case_np, axis=-1)  # shape (512,512,3)

        # Crop according to mask
        cropped_case, cropped_mask, crop_meta = roi_crop(case_np, mask_np, points=None)

        # Pad to fixed size (e.g., 512x512)
        padded_case, pad_meta = pad_or_resize_to_shape(cropped_case, shape_pad)
        
        padded_case = padded_case / 255.0
        
        padded_case = torch.from_numpy(padded_case.astype(np.float32)).permute(2, 0, 1)
        
        return padded_case, crop_meta, pad_meta

def convert_points_back(keypoints, crop_meta, pad_meta, shape_pad):
    """
    Converts the keypoints from the cropped image to the real image.
    """
    adjusted_points = keypoints[0][:,0:2].clone()
    
    # Denormalize
    adjusted_points[:, 0] += 0.5
    adjusted_points[:, 1] += 0.5

    adjusted_points[:, 0] *= shape_pad[0]
    adjusted_points[:, 1] *= shape_pad[1]

    if pad_meta['mode'] == 'pad' or pad_meta['mode'] == ['pad']:
        adjusted_points[:, 0] -= pad_meta['pad_left']
        adjusted_points[:, 1] -= pad_meta['pad_top']
    else:
        adjusted_points[:, 0] /= pad_meta['scale_x']
        adjusted_points[:, 1] /= pad_meta['scale_y']

    adjusted_points[:, 0] += crop_meta['x1']
    adjusted_points[:, 1] += crop_meta['y1']
    return adjusted_points

def get_biggest_object(mask_sitk):
    mask_sitk = sitk.GetImageFromArray(mask_sitk)

    # Connected component labeling
    cc = sitk.ConnectedComponent(mask_sitk)
    
    # Get statistics to find largest object
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    # Find label with the largest size
    largest_label = None
    largest_size = 0
    for label in stats.GetLabels():
        size = stats.GetPhysicalSize(label)
        if size > largest_size:
            largest_size = size
            largest_label = label
    
    # Create a binary mask with only the largest object
    largest_object = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    
    # Convert back to NumPy array
    return sitk.GetArrayFromImage(largest_object)


def get_biggest_and_neighbors(mask_np, distance_threshold=30, min_size=10):
    """
    Always keeps the largest connected component, and includes nearby components 
    within a spatial distance threshold that are at least `min_size` in size.

    Parameters:
    - mask_np: input binary mask as NumPy array (0s and 1s)
    - distance_threshold: max Euclidean distance (in physical units) to include other components
    - min_size: minimum physical size (in mm³) for nearby components

    Returns:
    - A binary NumPy array with the largest and qualifying neighboring components.
    """
    # Convert to SimpleITK image
    mask_sitk = sitk.GetImageFromArray(mask_np)
    
    # Label connected components
    cc = sitk.ConnectedComponent(mask_sitk)

    # Compute shape statistics
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    labels = stats.GetLabels()
    if not labels:
        return np.zeros_like(mask_np)

    # Always find the largest object regardless of size
    largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
    largest_centroid = np.array(stats.GetCentroid(largest_label))

    # Output image
    selected = sitk.Image(cc.GetSize(), sitk.sitkUInt8)
    selected.CopyInformation(cc)

    for label in labels:
        size = stats.GetPhysicalSize(label)
        centroid = np.array(stats.GetCentroid(label))

        if label == largest_label:
            selected = selected | sitk.Equal(cc, label)
        elif size >= min_size:
            distance = np.linalg.norm(centroid - largest_centroid)
            if distance <= distance_threshold:
                selected = selected | sitk.Equal(cc, label)

    return sitk.GetArrayFromImage(selected)


def remove_small_objects(mask_np, min_size=100):
    """
    Keeps all connected components larger than min_size (in physical units),
    and removes all smaller components.

    Parameters:
    - mask_np: input binary mask as NumPy array (0s and 1s)
    - min_size: minimum physical size (e.g., in mm^3) to keep a component

    Returns:
    - A binary NumPy array with only components >= min_size
    """
    # Convert to SimpleITK image
    mask_sitk = sitk.GetImageFromArray(mask_np)

    # Label connected components
    cc = sitk.ConnectedComponent(mask_sitk)

    # Compute shape statistics
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    # Output image
    filtered = sitk.Image(cc.GetSize(), sitk.sitkUInt8)
    filtered.CopyInformation(cc)

    # Keep components above min_size
    for label in stats.GetLabels():
        if stats.GetPhysicalSize(label) >= min_size:
            filtered = filtered | sitk.Equal(cc, label)

    return sitk.GetArrayFromImage(filtered)

### extract radiomic features for a case
# Helper function to extract features
def extract_features(image_mask_pair, extractor):
    ct_img, mask_img = image_mask_pair
    result = extractor.execute(ct_img, mask_img)
    return result

def run_parallel_extraction(image_mask_list, extractor, max_workers=64):
    # Submit each extraction job along with its associated input case
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [(item, executor.submit(extract_features, item, extractor)) for item in image_mask_list]
    
    feature_data = []
    for item, future in futures:
        result_dict = future.result()  # This will block until the future is done and return the OrderedDict
        feature_data.append(result_dict)
    
    return feature_data

def gray_img(img):
    """
    Return a 2D NumPy array (grayscale).
    """
    if (type(img) != np.ndarray):
        img = np.array(img)  # shape (H, W, 3)
    if img.ndim == 3 and img.shape[-1] == 3:
        # convert to uint8 grayscale by luminance (better than simple mean)
        # using integer ops then cast to float32 for radiomics if needed
        r, g, b = img[...,0], img[...,1], img[...,2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)  # still float
        # keep float32 for radiomics (radiomics accepts float images)
        return gray.astype(np.float32)
    elif img.ndim == 2:
        return img.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

def get_array(path):
    """
    Load an image from path and return a 2D NumPy array (grayscale).
    """
    img = Image.open(path).convert("RGB")
    return gray_img(img)

def load_npy_file(file_path):
    try:
        return np.load(file_path, allow_pickle=False)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise

def ensure_mask_stack_layout(masks, image_shape):
    """
    Accepts masks as np.ndarray and returns a list of 2D masks (H,W),
    ensuring they match the image_shape (H,W). Detects common layout possibilities:
      - (N, H, W) -> return masks[i]
      - (H, W, N) -> return masks[..., i]
      - (H, W) -> single mask
    If shapes don't match, tries transposition, otherwise raises.
    """
    H, W = image_shape
    masks = np.asarray(masks)
    if masks.ndim == 3:
        if masks.shape[1] == H and masks.shape[2] == W:
            # (N, H, W)
            return [masks[i].astype(np.uint8) for i in range(masks.shape[0])]
        if masks.shape[0] == H and masks.shape[1] == W:
            # (H, W, N)
            return [masks[..., i].astype(np.uint8) for i in range(masks.shape[2])]
        # other possibilities: (W, H, N) or (N, W, H) -> try to find a match by transposing
        # try all permutations to find (H,W)
        for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            permuted = np.transpose(masks, perm)
            if permuted.ndim == 3 and permuted.shape[1] == H and permuted.shape[2] == W:
                return [permuted[i].astype(np.uint8) for i in range(permuted.shape[0])]
        raise ValueError(f"Unable to interpret 3D mask array shape {masks.shape} to match image {H}x{W}")
    elif masks.ndim == 2:
        if masks.shape == (H,W) or masks.shape == (W,H):
            if masks.shape == (W,H):
                masks = masks.T
            return [masks.astype(np.uint8)]
        else:
            raise ValueError(f"2D mask shape {masks.shape} doesn't match image shape {image_shape}")
    else:
        raise ValueError("Mask array must be 2D or 3D")

def center_of_mass(mask: np.ndarray) -> Optional[Tuple[float, ...]]:
    """
    Compute center of mass of a NumPy mask and return coordinates in array-index order.
    Examples: 2D -> (row, col), 3D -> (z, row, col).

    Returns None if mask sum == 0.
    """
    mask = np.asarray(mask, dtype=float)
    if mask.size == 0:
        raise ValueError("empty mask provided")
    total = mask.sum()
    if total == 0:
        return None
    coords = np.indices(mask.shape, dtype=float)
    com = tuple(((coords[i] * mask).sum() / total) for i in range(mask.ndim))
    return com #(y,x)

#### Point detection and classification
CLASS_NAMES_INV = {
    "Left Hand": 0,
    "Right Hand": 1,
    "Scissors": 2,
    "Tweezers": 3,
    "Needle_Holder": 4,
    "Needle": 5
    }

num_keypoints_dict = {
    "Left Hand": 6,
    "Right Hand": 6,
    "Scissors": 3,
    "Tweezers": 3,
    "Needle_Holder": 3,
    "Needle": 3
    }

#### size util functions

def roi_crop(case_np, mask_np, points=None):
    """
    Crop case_np to the bounding box of non-zero pixels in mask_np.
    The `points` argument is accepted for API compatibility but is ignored.

    Args:
        case_np: np.ndarray - input image (H, W, C...) 
        mask_np: np.ndarray - single-channel grayscale/binary mask
        points: ignored (kept for compatibility)

    Returns:
        cropped_case, cropped_mask, adjusted_points, meta
        adjusted_points will be None (points are ignored)
    """
    # find non-zero mask coords
    coords = cv2.findNonZero(mask_np)

    if coords is None:
        # If mask is empty we cannot compute ROI (points are ignored)
        raise ValueError("Mask is empty/invalid; cannot compute ROI when points are ignored.")

    x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(coords)
    mask_bbox = (int(x_mask), int(y_mask), int(x_mask + w_mask), int(y_mask + h_mask))

    # union = mask_bbox (points ignored)
    union_minx, union_miny, union_maxx, union_maxy = mask_bbox

    # padding (pixels)
    padding = 50

    # Expand bounding box (clamp to image bounds)
    x1 = max(0, int(union_minx - padding))
    y1 = max(0, int(union_miny - padding))
    x2 = min(case_np.shape[1], int(union_maxx + padding))
    y2 = min(case_np.shape[0], int(union_maxy + padding))

    # Crop image and mask
    cropped_case = case_np[y1:y2, x1:x2].copy()
    cropped_mask = mask_np[y1:y2, x1:x2].copy()

    meta = {
        'mode': 'crop',
        'x1': int(x1),
        'y1': int(y1),
        'x2': int(x2),
        'y2': int(y2),
        'padding': int(padding),
    }

    return cropped_case, cropped_mask, meta

def pad_or_resize_to_shape(case_np: np.ndarray,
                           shape: Tuple[int, int] = (256, 256)
                           ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Resize (if larger) or pad (if smaller) `case_np` to `shape`.
    The `points` argument is accepted for compatibility but is ignored.
    Returns (processed_case, adjusted_points, meta) where adjusted_points is always None.
    """
    target_h, target_w = shape
    current_h, current_w = case_np.shape[:2]

    processed_case = case_np.copy()
    meta = {}

    # Resize if either dimension is larger than target
    if current_h > target_h or current_w > target_w:
        scale_x = target_w / float(current_w)
        scale_y = target_h / float(current_h)
        processed_case = cv2.resize(processed_case, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        meta = {
            'mode': 'resize',
            'scale_x': float(scale_x),
            'scale_y': float(scale_y),
            'original_shape': (int(current_h), int(current_w)),
            'target_shape': (int(target_h), int(target_w)),
        }
        return processed_case, meta

    # Pad if smaller (or equal -> zero padding)
    pad_h_total = max(0, target_h - current_h)
    pad_w_total = max(0, target_w - current_w)
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    # Build pad widths for np.pad: (H, W, [C])
    pad_widths = [(pad_top, pad_bottom), (pad_left, pad_right)]
    for _ in range(processed_case.ndim - 2):
        pad_widths.append((0, 0))

    if any(p != 0 for pair in pad_widths for p in pair):
        processed_case = np.pad(processed_case, pad_widths, mode='constant', constant_values=0)

    meta = {
        'mode': 'pad',
        'pad_top': int(pad_top),
        'pad_bottom': int(pad_bottom),
        'pad_left': int(pad_left),
        'pad_right': int(pad_right),
        'original_shape': (int(current_h), int(current_w)),
        'target_shape': (int(target_h), int(target_w)),
    }
    return processed_case, meta

def flatten_entry(entry):
    flat_values = []
    for x in entry:
        if isinstance(x, torch.Tensor):
            flat_values.extend(x.tolist())   # convert tensor to Python numbers
        elif isinstance(x, np.generic):
            flat_values.append(float(x))     # convert numpy scalar to float
        else:
            flat_values.append(x)            # keep ints/floats
    return flat_values

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    #bb_x, bb_y, bb_w, bb_h
    return cmin, rmin, cmax-cmin, rmax-rmin

if __name__ == '__main__':
    pass