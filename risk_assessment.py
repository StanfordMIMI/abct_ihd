"""
IHD Risk Assessment (prediction) for a given cohort using trained models
Performs 1 and 5y IHD risk assessment using:
    -L3 slice only
    -Clinical data only
    -Imaging + Clinical Fusion
"""

import argparse
import os
import configparser
import torch
from torch.utils.data import DataLoader
import ast
from importlib import import_module
import pandas as pd
from torch.nn.functional import softmax
risk_img_models = import_module('imaging.2_IHD_prediction.models.risk_detector_models')
ihd_dataloader = import_module('imaging.2_IHD_prediction.utils.dataset_dataloader')
data_transforms = import_module('imaging.2_IHD_prediction.utils.data_transforms')
import numpy as np
import joblib

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def parse_args():
    """Parses command line arguments (path to config file)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-risk_assessment", help="whether to make 'I' (image only), 'C' (clinical only) or 'both' (fusion) predictions",
                        default = 'both', type=str)
    parser.add_argument("-trained_model_dir", help="path to trained model .h5 files",
                        default='./models/', type=str)
    parser.add_argument("-image_data_dir", help="path to L3 slice .dcm files",
                        default='./data/images/l3_slices/', type=str)
    parser.add_argument("-clinical_data_path", help="path to clinical data .csv files",
                        default='./data/clinical/clinical_data.csv', type=str)
    parser.add_argument("-output_dir", help="path to output dir where predictions are saved",
                        default = './predictions/', type=str)
    args = parser.parse_args()
    # logging.basicConfig(filename=args.log, level=logging.INFO)
    return args.risk_assessment, args.trained_model_dir, args.image_data_dir, \
            args.clinical_data_path, args.output_dir

def load_model_config(config_f_path):
    """Returns a dictionary with a configuration from a file"""
    config = configparser.ConfigParser()
    config.read(config_f_path)
    
    config = [dict(config.items(s)) for s in config.sections()][0]

    convert2int = ['num_neighbors', 'data_loader_workers', 'seed', 'batch_size']
    for c2i in convert2int:
        if c2i in config:
            config[c2i] = int(config[c2i])

    convert2float = ['thresh_low','thresh_high','new_spacing']
    for c2f in convert2float:
        if c2f in config:
            config[c2f] = float(config[c2f])

    convert2lit = ['new_img_size','shuffle','label_mapper']
    for lit in convert2lit:
        if lit in config:
            config[lit] = ast.literal_eval(config[lit])

    convert2bool = ['preload_all_data', 'normalize_fat', 'normalize_spacing','frozenweights',\
                    'remove_bed', 'as_integer', 'pretrained']

    for boo in convert2bool:
        if boo in config:
            config[boo] = ast.literal_eval(config[boo])

    return config

def load_models(risk_assessment, trained_model_dir, device=torch.device('cpu')):
    models = {}
    if risk_assessment == 'I' or risk_assessment == 'both':
        def get_model(weight_path, config_path):
            config = load_model_config(config_path)
            config['pretrained'] = False
            model = risk_img_models.efficientNet(config)
            model_weights = torch.load(weight_path, map_location = device)
            model.load_state_dict(model_weights)
            model.eval()
            return model

        best_1y_image_only_model_path = os.path.join(trained_model_dir,'L3_IHD_efficientnet_1y.pth')
        best_1y_image_only_config_path = os.path.join(trained_model_dir,'L3_IHD_efficientnet_1y.cfg')
        best_5y_image_only_model_path = os.path.join(trained_model_dir,'L3_IHD_efficientnet_5y.pth')
        best_5y_image_only_config_path = os.path.join(trained_model_dir,'L3_IHD_efficientnet_5y.cfg')

        models['img_1y'] = get_model(best_1y_image_only_model_path, best_1y_image_only_config_path)
        models['img_5y'] = get_model(best_5y_image_only_model_path, best_5y_image_only_config_path)
        
    if risk_assessment == 'C' or risk_assessment == 'both':
        best_5y_clin_only_model_path = os.path.join(trained_model_dir,'clinical_5y.pkl')
        best_1y_clin_only_model_path = os.path.join(trained_model_dir,'clinical_1y.pkl')
        models['clin_1y'] = joblib.load(best_1y_clin_only_model_path)
        models['clin_5y'] = joblib.load(best_5y_clin_only_model_path)
        
    if risk_assessment == 'both':
        best_5y_ICfusion_model_path = os.path.join(trained_model_dir,'img_clin_fusion_5y.pkl')
        best_1y_ICfusion_model_path = os.path.join(trained_model_dir,'img_clin_fusion_1y.pkl')
        models['ICfusion_1y'] = joblib.load(best_1y_ICfusion_model_path)
        models['ICfusion_5y'] = joblib.load(best_5y_ICfusion_model_path)
        
    return models

def load_image_data(image_data_dir, trained_model_dir):
    best_image_only_config_path = os.path.join(trained_model_dir,'L3_IHD_efficientnet_1y.cfg')
    config = load_model_config(best_image_only_config_path)
    b_size = config['batch_size']//2
    data_shuffle = config['shuffle']
    n_workers = config['data_loader_workers']
    dataset = ihd_dataloader.dcmDataset(config, image_data_dir)
    dataset.setTransform(data_transforms.getTransform(config['data_transforms_val']))

    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=data_shuffle, num_workers=n_workers)

    return dataloader
def load_clinical_data(clinical_data_path, desired_cols, id_col='anon_id'):
    data = pd.read_csv(clinical_data_path)
    if 'age' in data.columns:
        data = data.rename(columns={'age':'age_at_scan'})
    if 'sex' in data.columns:
        data['gender'] =  data.apply(lambda x: 0 if 'female' in x['sex'] else 1, axis=1)
    return data[id_col], data[desired_cols]

def get_img_preds(models, image_data, device):
    preds = {}
    models['img_1y'].to(device)
    models['img_5y'].to(device)
    
    for x,keys in image_data:
        x=x.to(device)    
        preds_1y = softmax(models['img_1y'](x), dim=1).detach()[:,1].tolist()
        
        preds_5y = softmax(models['img_5y'](x), dim=1).detach()[:,1].tolist()
        
        for k, pred1y, pred5y in zip(keys, preds_1y, preds_5y):
            if k not in preds:
                preds[k] = {}
            
            preds[k]['img_1y'] = pred1y
            preds[k]['img_5y'] = pred5y
    return preds

def get_clin_preds(clinical_data_path, models):
    ids, clin_data_1y = load_clinical_data(clinical_data_path, models['clin_1y'].get_booster().feature_names)
    
    _, clin_data_5y = load_clinical_data(clinical_data_path, models['clin_5y'].get_booster().feature_names)
    
    clin_1y_preds = models['clin_1y'].predict_proba(clin_data_1y)[:,1].tolist()
    clin_5y_preds = models['clin_5y'].predict_proba(clin_data_5y)[:,1].tolist()

    return {str(k):{'clin_1y':p1, 'clin_5y':p2} for k,p1,p2 in zip(ids, clin_1y_preds, clin_5y_preds)}

def get_fusion_preds(preds, models):
    for k in preds:
        clin_pred_1y = preds[k]['clin_1y']
        clin_pred_5y = preds[k]['clin_5y']
        img_pred_1y = preds[k]['img_1y']
        img_pred_5y = preds[k]['img_5y']
        
        preds[k]['I_C_fusion_1y'] = models['ICfusion_1y']['clf'].predict_proba([[clin_pred_1y, img_pred_1y]])[:,1].tolist()[0]
        preds[k]['I_C_fusion_5y'] = models['ICfusion_5y']['clf'].predict_proba([[clin_pred_5y, img_pred_5y]])[:,1].tolist()[0]

    return preds
def main():
    risk_assessment, trained_model_dir, image_data_dir, clinical_data_path, out_dir = parse_args()
    device = get_device()
    
    models = load_models(risk_assessment, trained_model_dir, device)
    if risk_assessment=='I' or risk_assessment=='both':
        ## IHD Risk Assessment using L3 slice
        image_data = load_image_data(image_data_dir, trained_model_dir)
        img_preds = get_img_preds(models, image_data, device)

    if risk_assessment=='C' or risk_assessment=='both':
        ## IHD Risk Assessment using L3 slice
        clin_preds = get_clin_preds(clinical_data_path, models)
        
    if risk_assessment=='both':
        preds = {k:{**v, **clin_preds[k]} for k,v in img_preds.items()}
        preds = get_fusion_preds(preds, models)
    if risk_assessment=='I':
        preds = img_preds
    elif risk_assessment=='C':    
        preds = clin_preds
    
    pd.DataFrame(preds).transpose().to_csv(os.path.join(out_dir, f'predictions_{risk_assessment}.csv'))
           
if __name__ == '__main__':
    main()
