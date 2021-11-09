"""
Evaluate best models and save predictions to output file
"""
import torch
import numpy as np
import torch.nn.functional as F
from utils.dataset_dataloader import h5Dataset
from utils.dataset_dataloader import trainValTestSplit
from utils.data_transforms import getTransform
from utils.model_eval import getAUROC
from train import load_config, select_model, get_dataloaders

if __name__ == "__main__":
    
    #best 5 year outcome cohort

    modelPath = '/PATH_TO/models/efficientnet_5y_1_0lr_6e-06_batchSize_8_nEpochs_9_auroc_0.7947.pth' 
    config_path = './configs/efficientnet_5y_1_0.cfg'
    transformed_data_path = '/PATH_TO/predictions/5y_image_only_preds.csv' 
        
    #best 1 year outcome cohort

    modelPath = '/PATH_TO/models/efficientnet_1y_1_0lr_7e-06_batchSize_8_nEpochs_7_auroc_0.7733.pth'
    configPath = './configs/efficientnet_1y_1_0.cfg'
    transformed_data_path = '/PATH_TO/predictions/1y_image_only_preds.csv' 

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    config = load_config(config_path) 

    #Prepare data
    dset = h5Dataset(config, transform = None)

    train_dataset, val_dataset, test_dataset = trainValTestSplit(dset, config['split_file'])

    train_dataset.dataset.setTransform(getTransform(config['data_transforms_train']))
    val_dataset.dataset.setTransform(getTransform(config['data_transforms_val']))
    test_dataset.dataset.setTransform(getTransform(config['data_transforms_val']))

    dataset_sizes = {'train': len(train_dataset), 'val':len(val_dataset),
                    'test':len(test_dataset)}

    config['shuffle'] = False
    dataloaders = get_dataloaders(train_dataset, val_dataset, test_dataset, config)    
    
    model = select_model(config)
    trained_model = torch.load(modelPath, map_location = device)
    model.load_state_dict(trained_model)
    model.to(device)
    curve={}
    AUROC = {}
    epoch_loss={}
    epoch_acc = {}

    with open(transformed_data_path,'w') as fout:
        fout.write('\t'.join(['anon_id','score_0', 'score_1'])+'\n')

    for phase in ['train','val','test']:
        model.eval()  # Set model to evaluate mode
        running_corrects = 0
        running_labels = []
        running_preds = []
        running_loss = 0.0
        running_keys = []
        running_scores = []
        running_scores0 = []
        running_scores1 = []
        running_probs=[]
        for t, (x, y, k) in enumerate(dataloaders[phase]):
            x = x.float().to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            with torch.set_grad_enabled(False):
                scores=model(x)
                probs = F.softmax(scores, dim=1)[:,1]
                _, preds = torch.max(scores, 1)
                loss = F.binary_cross_entropy_with_logits(scores, y) 

             # statistics
            running_loss += loss.item() * x.size(0)
            for key in k:
                running_keys.append(key)
            running_corrects += torch.sum(preds==torch.max(y, 1).indices)
            running_labels.extend(list(np.argmax(y.data.cpu().numpy(), axis=1)))
            running_scores.extend(list(scores.cpu().numpy()))
            running_scores0.extend(list(scores.cpu().numpy()[:,0]))
            running_scores1.extend(list(scores.cpu().numpy()[:,1]))
            running_preds.extend(list(preds.cpu().numpy()))
            running_probs.extend(list(probs.cpu().numpy()))
        with open(transformed_data_path,'a') as fout:
            for i,k in enumerate(running_keys):
                fout.write('\t'.join([str(x) for x in [k, running_scores[i][0], running_scores[i][1]]])+'\n')
        epoch_loss[phase] = running_loss / dataset_sizes[phase]
        epoch_acc[phase] = running_corrects.double() / dataset_sizes[phase]
        curve[phase], AUROC[phase] = getAUROC(np.array(running_labels), np.array(running_probs))
        print(f"{phase} \n\t AUROC: {AUROC[phase]['AUROC']:.3f}")
