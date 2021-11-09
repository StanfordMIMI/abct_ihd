"""
Calculate pixel-wise saliency for images in dataset using trained models
"""
import argparse
import torch
import h5py as h5
from utils.dataset_dataloader import trainValTestSplit, h5Dataset
from utils.data_transforms import getTransform
from train import load_config, select_model, get_dataloaders
from captum.attr import Saliency
import logging 

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using gpu")
else:
    device = torch.device('cpu')

def parse_args():
    """Parses command line arguments (path to config file)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", help="type of attribution",
                        default = 'saliency')
    parser.add_argument("-cohort", help="suffix to add to output file path (i.e. after /dataNAS/people/jmz/data/opportunistic/abct_ihd/attributions/)",
                        default = '1y_')
    
    args = parser.parse_args()
    
    return str(args.type), str(args.cohort)

def load_model_and_data(modelPath, configPath):

    config = load_config(configPath)

    dset = h5Dataset(config, transform = None)

    train_dataset, val_dataset, test_dataset = trainValTestSplit(dset, config['split_file'])

    train_dataset.dataset.setTransform(getTransform(config['data_transforms_train']))
    val_dataset.dataset.setTransform(getTransform(config['data_transforms_val']))
    test_dataset.dataset.setTransform(getTransform(config['data_transforms_val']))
    config['batch_size'] = 4
    config['shuffle'] = False
    dataloaders = get_dataloaders(train_dataset, val_dataset, test_dataset, config)
    model = select_model(config)
    trained_model = torch.load(modelPath, map_location = device)
    model.load_state_dict(trained_model)

    return model, dataloaders, config

def get_saliency_vals(model, dataloaders, out_path='/PATH_TO/data/2D_img_saliency.h5'):
    dtype = torch.float32
    model.zero_grad()
    model.to(device=device)
    saliency = Saliency(model)

    with h5.File(out_path, 'w') as h5out:
        for phase in ['train','val','test']:
            for _, (x, y, k) in enumerate(dataloaders[phase]):
                x = x.float().to(device=device, dtype=dtype).requires_grad_(requires_grad=True)
                y = y.to(device=device, dtype=dtype)

                slncy = saliency.attribute(x, target=1, abs=False)

                slncy = slncy.detach().cpu().numpy()
                for i,key in enumerate(k):
                    h5out.create_dataset(key, (528,528), data=slncy[i,1,:,:])
    return

def main():
    attr_type, outPathSuffix = parse_args()

    if outPathSuffix == '1y_':
        # best 1 year outcome cohort model
        modelPath = '/PATH_TO/models/efficientnet_1y_1_0lr_7e-06_batchSize_8_nEpochs_7_auroc_0.7733.pth'
        configPath = './configs/efficientnet_1y_1_0.cfg'
    elif outPathSuffix == '5y_':
        # best 5 year outcome cohort model
        modelPath = modelPath = '/PATH_TO/models/efficientnet_5y_1_0lr_6e-06_batchSize_8_nEpochs_9_auroc_0.7947.pth'
        configPath = './configs/efficientnet_5y_1_0.cfg'
    else:
        raise ValueError(f"invalid model selected: {outPathSuffix}")

    model, dataloaders, _ = load_model_and_data(modelPath, configPath)
    logging.info("Loaded model and data")
    outPathBase = '/PATH_TO/data/'

    if attr_type == "saliency":
        get_saliency_vals(model, dataloaders, outPathBase + outPathSuffix + attr_type + '.h5')
    else:
        logging.error("Attribution type not found")

if __name__=='__main__':
    main()
