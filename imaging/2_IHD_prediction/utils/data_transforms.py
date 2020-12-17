import torchvision.transforms as transforms
import torch.nn.functional as F

def getTransform(transformType):
    if transformType == 'train_basic_efficientb6':
        return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.unsqueeze(dim=0)),
                                transforms.Lambda(lambda x: F.interpolate(x, size=528)),
                                transforms.Lambda(lambda x: x.squeeze(dim=0)),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
    elif transformType == 'train_v2_efficientb6':
        return transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomAffine(degrees=3, translate=(.005,.005)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.unsqueeze(dim=0)),
                                transforms.Lambda(lambda x: F.interpolate(x, size=528)),
                                transforms.Lambda(lambda x: x.squeeze(dim=0)),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
    elif transformType == "val_basic_efficientb6":
        return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.unsqueeze(dim=0)),
                                transforms.Lambda(lambda x: F.interpolate(x, size=528)),
                                transforms.Lambda(lambda x: x.squeeze(dim=0)),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
    else:
        print("incorect transform type specified")
        exit()
