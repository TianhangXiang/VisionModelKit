import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def imagenet(args):
    data_dir = args.data_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    train_data_dir = os.path.join(data_dir, "train")
    train_dataset = datasets.ImageFolder(
                train_data_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
    
    val_data_dir = os.path.join(data_dir, "val")
    val_dataset = datasets.ImageFolder(
                val_data_dir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
    
    collate_fn = None

    return train_dataset, val_dataset, collate_fn