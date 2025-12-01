import os
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_data_transforms():
    """
    Returns the data transformations for training and testing.
    Standardizes input to 224x224 for ViT and ResNet models.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def download_dataset():
    """
    Downloads the CIFAKE dataset using kagglehub.
    Returns the path to the dataset.
    """
    print("Downloading/Verifying CIFAKE dataset...")
    path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    print(f"Dataset path: {path}")
    return path

def get_dataloaders(data_dir, batch_size=128):
    """
    Creates dataloaders for the CIFAKE dataset.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the dataloaders.
        
    Returns:
        dataloaders (dict): Dictionary containing 'train' and 'test' DataLoaders.
        dataset_sizes (dict): Dictionary containing sizes of 'train' and 'test' datasets.
        class_names (list): List of class names.
    """
    data_transforms = get_data_transforms()
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    workers = os.cpu_count() or 2
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=workers)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names
