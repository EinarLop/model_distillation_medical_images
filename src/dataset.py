import io
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CheXpertZipDataset(Dataset):
    def __init__(self, zip_path, csv_internal_path, processor):
        """
        Data pre-processing for CheXpert dataset
        Args:
            zip_path: Path to CheXpert.zip file
            csv_internal_path: Path to the CSV inside the zip 
            processor: AutoImageProcessor from the teacher model
        """
        self.zip_path = zip_path
        self.processor = processor
        
        # Open the zip to extract cvs
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(csv_internal_path) as f:
                self.df = pd.read_csv(f)
                
        # 5 classes expected by the'codewithdark/vit-chest-xray' teacher
        self.target_columns = [
            'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding'
        ]
        
        # Fill blanks (NaN) with 0
        # Map uncertain labels (-1) to 0 for the baseline
        self.df[self.target_columns] = self.df[self.target_columns].fillna(0).replace(-1, 0)
        
        # Store paths and labels in memory for fast indexing
        self.image_paths = self.df['Path'].values
        self.labels = self.df[self.target_columns].values.astype('float32')
        
        print(f"Dataset pre-processed with {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get single image form dataset, without extracting zip
        """
        img_path = self.image_paths[idx]
        img_path = img_path.replace("CheXpert-v1.0-small/", "")

        label_tensor = torch.tensor(self.labels[idx])
        
        # Open the zip archive on the fly for each image
        # Here instead of __init__ because PyTorch multiprocessing workers will crash if they try to share a single open file connection
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(img_path) as f:
                img_bytes = f.read()
                
        # Raw bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Pass through teacher's processor
        # Automatically resizing to 224x224 and normalization
        processed = self.processor(images=image, return_tensors="pt")
        
        # The processor adds a batch dimension [1, C, H, W], so we squeeze it out to [C, H, W]
        pixel_values = processed['pixel_values'].squeeze(0)
        
        return pixel_values, label_tensor