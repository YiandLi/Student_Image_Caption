import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import train_test_split
from data.edu_karpathy_dataset import edu_karpathy
from transform.randaugment import RandomAugment


def get_annotation(eval_ratio):
    dess = pd.read_csv("data/descriptions.csv")
    # print(dess.head())
    dess = [{"caption": des, "image_id": img_path}
            for des, img_path in
            zip(dess['description'].tolist(), dess['file'].tolist())]
    
    train_dess, dev_dess = train_test_split(dess, test_size=eval_ratio, random_state=42)
    print(f"\nTrain set length: {len(train_dess)}, Dev set length: {len(dev_dess)}\n")
    
    return train_dess, dev_dess


def create_dataset(config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_annotation, eval_annotation = get_annotation(config['eval_ratio'])
    train_dataset = edu_karpathy(transform_train, train_annotation, prompt=config['prompt'])
    val_dataset = edu_karpathy(transform_test, eval_annotation, prompt="")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    return train_loader, val_loader
