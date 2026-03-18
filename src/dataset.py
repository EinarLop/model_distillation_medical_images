import io
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CheXpertDataset(Dataset):
    def __init__(self, data_root, csv_path, processor):
        """
        Args:
            data_root (str): The base directory where the dataset was extracted.
            csv_path (str): The full path to the train.csv or valid.csv file.
            processor: The Hugging Face processor.
        """
        self.data_root = data_root
        self.processor = processor
        
        print(f"Reading CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
                
        self.target_columns = [
            'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding'
        ]
        
        self.df[self.target_columns] = self.df[self.target_columns].fillna(0).replace(-1, 0)
        
        self.image_paths = self.df['Path'].values
        self.labels = self.df[self.target_columns].values.astype('float32')
        
        print(f"Dataset initialized with {len(self.df)} images across 5 classes.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Join the root directory with the relative path from the CSV
        full_img_path = os.path.join(self.data_root, self.image_paths[idx])
        full_img_path = full_img_path.replace("CheXpert-v1.0-small/", "")
        label_tensor = torch.tensor(self.labels[idx])
        
        # Open directly from the disk
        image = Image.open(full_img_path).convert('RGB')
        
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        
        return pixel_values, label_tensor