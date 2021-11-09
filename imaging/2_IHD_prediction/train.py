"""
Train efficientnet IHD prediction model
"""
import argparse
import time
import copy
import configparser
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset_dataloader import h5Dataset
from models import risk_detector_models
from utils.dataset_dataloader import trainValTestSplit
from utils.data_transforms import getTransform
from utils.model_eval import getAUROC
from utils.loss import FocalLoss
import logging

def parse_args():
    """Parses command line arguments (path to config file)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="path to config file",
                        default = './configs/sample_config.cfg')
    parser.add_argument("-log", help="path to log dir",
                            default = '')
    args = parser.parse_args()
    logging.basicConfig(filename=str(args.log)+str(args.config).replace('configs','logs').replace('.cfg','.log'), level=logging.INFO)
    return str(args.config)

def load_config(config_f_path):
    """Returns a dictionary with a configuration from a file"""
    config = configparser.ConfigParser()
    config.read(config_f_path)
    config = [dict(config.items(s)) for s in config.sections()][0]

    convert2int = ['num_neighbors', 'max_epoch','batch_size','data_loader_workers', 'seed']
    for c2i in convert2int:
        if c2i in config:
            config[c2i] = int(config[c2i])

    convert2float = ['thresh_low','thresh_high','new_spacing','learning_rate',\
                    'learn_rate_base','focal_loss_gamma']
    for c2f in convert2float:
        if c2f in config:
            config[c2f] = float(config[c2f])

    convert2lit = ['new_img_size','shuffle','label_mapper','focal_loss_alpha']
    for lit in convert2lit:
        if lit in config:
            config[lit] = ast.literal_eval(config[lit])

    convert2bool = ['preload_all_data', 'normalize_fat', 'normalize_spacing','frozenweights',\
                    'remove_bed', 'as_integer', 'differential_lr','use_specific_subset', \
                    'focal_loss']

    for boo in convert2bool:
        if boo in config:
            config[boo] = ast.literal_eval(config[boo])

    return config

def select_model(config):
    """Returns a model object from riskdetectormodels"""
    model_name = config['model_name']
    if model_name =="efficientnet":
        return risk_detector_models.efficientNet(config)
    else:
        print("Incorrect model type specification")
        return

def get_dataloaders(train_dataset, val_dataset, test_dataset, config):
    """
    Returns a dictionary containing train/val/test keys
    mapping to their respecting dataloaders
    """
    b_size = config['batch_size']
    data_shuffle = config['shuffle']
    n_workers = config['data_loader_workers']
    train_dl = DataLoader(train_dataset, batch_size=b_size, shuffle=data_shuffle, num_workers=n_workers)
    val_dl = DataLoader(val_dataset, batch_size=b_size, shuffle=data_shuffle, num_workers=n_workers)
    test_dl = DataLoader(test_dataset, batch_size=b_size, shuffle=data_shuffle, num_workers=n_workers)

    return {'train':train_dl, 'val':val_dl, 'test':test_dl}

def save_checkpoint(config, model, best_val_auroc):
    """
    Saves a checkpoint
    Params:
    filename: name of checkpoint file
    """
    filename = config['model_path'] + config['model_name'] + '/' + config['exp_name']+ 'lr_' + \
                str(config['learning_rate']) + '_batchSize_' + str(config['batch_size']) + \
                '_nEpochs_' + str(config['max_epoch']) + '_auroc_' + f'{best_val_auroc:.4f}' + '.pth'
    torch.save(model.state_dict(), filename)
    return

#Training and validation
def train_one_epoch(scheduler, model, device, dtype, dataloaders, optimizer, criterion,\
                    dataset_sizes, verbose=False):
    """
    Performs one epoch of training
    """
    AUROC = {}
    epoch_loss={}
    epoch_acc = {}

    for phase in ['train', 'val']:
        if phase == 'train':
            if scheduler is not None:
                scheduler.step()
            model.train()
        else:
            model.eval()

        running_corrects = 0
        running_labels = []
        running_probs = []
        running_loss = 0.0

        for t, (x, y, _) in enumerate(dataloaders[phase]):
            x = x.float().to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs=model(x)
                probs = F.softmax(outputs, dim=1).detach()
                _, preds = torch.max(probs, 1)
                loss = criterion(outputs, torch.max(y, 1)[1])

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    if verbose and t%100 == 0:
                        logging.info('Iteration %d, loss = %.4f' % (t, loss.item()))
                        logging.info()

             # statistics
            running_loss += loss.item() * x.size(0)

            running_corrects += torch.sum(preds==torch.max(y, 1).indices)
            running_labels.extend(list(np.argmax(y.data.cpu().numpy(), axis=1)))
            running_probs.extend(list(probs.cpu().numpy()))
        epoch_loss[phase] = running_loss / dataset_sizes[phase]
        epoch_acc[phase] = running_corrects.double() / dataset_sizes[phase]
        _, AUROC[phase] = getAUROC(np.array(running_labels), np.array(running_probs)[:,1])

        logging.info(f"{phase} \t Loss: {epoch_loss[phase]:.4f} \t AUROC: {AUROC[phase]['AUROC']:.4f} \t Acc:{epoch_acc[phase]:.4f}")
    return model, epoch_loss, AUROC

def train_model(model, criterion, learn_rate, scheduler, num_epochs, dataloaders, dataset_sizes, verbose=False,
               differential_lr=False, learn_rate_base=None):
    """
    Model is trained. Best model and AUROC are returned.
    """
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') #not recommended
    logging.info(f"using {device}")
    since = time.time()
    
    model.to(device)
    if differential_lr == False:
        optimizer = optim.Adam(model.trainableParams(), lr=learn_rate)
    elif differential_lr == True:
        blocks, fc = model.differentialTrainableParams()
        optimizer = optim.Adam([{'params':blocks},
                       {'params':fc, 'lr':learn_rate}], lr=learn_rate_base)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auroc = 0.0

    for epoch in range(num_epochs):
        print(f'---------- \t Epoch {epoch}/{num_epochs-1} ----------')

        model, _, epoch_auroc = \
                train_one_epoch(scheduler, model, device, dtype, dataloaders, \
                                optimizer, criterion, dataset_sizes, verbose)

        if epoch_auroc['val']['AUROC'] > best_auroc:
            best_auroc = epoch_auroc['val']['AUROC']
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info(f'Saving model weights with best val AUROC: {best_auroc}')
    logging.info(f'Training complete in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_auroc

def main():
    config_path = parse_args()
    config = load_config(config_path)

    #Prepare data
    dset = h5Dataset(config, transform = None)

    train_dataset, val_dataset, test_dataset = trainValTestSplit(dset, config['split_file'])
    train_dataset.dataset.setTransform(getTransform(config['data_transforms_train']))
    val_dataset.dataset.setTransform(getTransform(config['data_transforms_val']))

    dataset_sizes = {'train': len(train_dataset), 'val':len(val_dataset)}

    dataloaders = get_dataloaders(train_dataset, val_dataset, test_dataset, config)

    #Prepare model
    model = select_model(config)
    if 'focal_loss' in config and config['focal_loss'] == True:
        if 'focal_loss_gamma' not in config:
            criterion = FocalLoss(alpha=torch.tensor(config['focal_loss_alpha']))
        else:
            criterion = FocalLoss(alpha=torch.tensor(config['focal_loss_alpha']), \
                                    gamma = config['focal_loss_gamma'])
    else:
        criterion = nn.CrossEntropyLoss()

    if 'differential_lr' in config:
        best_model, best_auroc = train_model(model, criterion, config['learning_rate'], None, \
                                            config['max_epoch'], dataloaders, dataset_sizes, \
                                            differential_lr=config['differential_lr'], \
                                            learn_rate_base=config['learn_rate_base'])
    else:
        best_model, best_auroc = train_model(model, criterion, config['learning_rate'], None, \
                                            config['max_epoch'], dataloaders, dataset_sizes)

    save_checkpoint(config, best_model, best_auroc)

if __name__ == "__main__":
    main()
