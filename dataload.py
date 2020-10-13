import numpy as np
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import string
from torch.utils.data import Dataset, DataLoader
import os


class CaptchaDataset(Dataset):
    def __init__(self, path, transform=None):
        assert os.path.exists(path), -1
        self.images = [os.path.join(path, img) for img in os.listdir(path)]
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        label = img_path.split('/')[-1][:5].upper()
        label = encode(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(label)


def get_transform(PATH, batch_size):
	transform = transforms.Compose([
	    # transforms.Resize(224),                         
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	data = CaptchaDataset(PATH, transform=transform)

	dataloader = torch.utils.data.DataLoader(
		data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
	return dataloader



def decode(pred):
    const = "ABCDEF" + string.digits
    lenght = len(const)
    targets = np.zeros((pred.shape[0], 5), dtype=str)
    for i in range(pred.shape[0]):
        c1 = const[torch.argmax(pred[i][:lenght])]
        c2 = const[torch.argmax(pred[i][lenght:2*lenght])]
        c3 = const[torch.argmax(pred[i][2*lenght:3*lenght])]
        c4 = const[torch.argmax(pred[i][3*lenght:4*lenght])]
        c5 = const[torch.argmax(pred[i][4*lenght:])]
        targets[i] = np.array([c1, c2, c3, c4, c5])
    return targets


def encode(label):
    const = "ABCDEF" + string.digits
    lenght = len(const)
    multi_labels = np.zeros((lenght * 5))
    for i, c in enumerate(label):
        multi_labels[i*lenght:(i+1)*lenght][const.index(c)] = 1
    return multi_labels