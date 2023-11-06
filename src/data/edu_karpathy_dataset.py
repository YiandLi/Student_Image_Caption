import os
from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption


# class coco_karpathy_train(Dataset):
class edu_karpathy(Dataset):
    def __init__(self, transform, annotation, max_words=50, prompt=''):
        self.annotation = annotation
        self.image_root = 'data/images'
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        
        self.img_ids = {}
        
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]
