import sys
sys.path.append('../')
from fusion.IHD_ml_models_2Dimg_ontology_uncorr_ensemble import *

def load_seg_data(section_path, label_col=''):

    computed_l3_metrics = pd.read_csv('/PATH_TO/data/IHD_8139_segmentation_fts.csv') #file containing BC metrics for all individuals
    section_data = pd.read_csv(section_path)[['anon_id',label_col,'set']]
    section_data = section_data.rename({label_col:'label'}, axis='columns')
    
    section_data = pd.merge(section_data, computed_l3_metrics, how='left', left_on='anon_id', right_on='anon_id')
    
    section_data_train = section_data[(section_data['set']=='train')|(section_data['set']=='val')].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)

    train_x = section_data_train[['muscle_HU', 'vat_sat_ratio']]
    train_y = section_data_train.label
    train_id = section_data_train.anon_id.values
    
    test_x = section_data_test[['muscle_HU', 'vat_sat_ratio']]
    test_y = section_data_test.label
    test_id = section_data_test.anon_id.values
    return {'train': {'X':train_x, 'y':train_y, 'id':train_id}, 'test': {'X':test_x, 'y':test_y, 'id':test_id}}
def get_seg_preds():
     ## Import segmentation metrics model
    model_dir = '/PATH_TO/models/segmentation_metrics/'
    best_seg_metrics_1y = joblib.load(model_dir+'best_segmentation_fts_Logistic1y_regression.pkl')
    best_seg_metrics_5y = joblib.load(model_dir+'best_segmentation_fts_Logistic5y_regression.pkl')

    data_path_1y = '/PATH_TO/data/IHD_8139_1y_train_val_test_split.csv'
    data_1y = load_seg_data(data_path_1y, '1y_label')
    data_path_5y = '/PATH_TO/data/IHD_8139_5y_train_val_test_split.csv'
    data_5y = load_seg_data(data_path_5y, '5y_label')

    preds_seg_1y_train = best_seg_metrics_1y.predict_proba(data_1y['train']['X'])
    
    preds_seg_1y_test = best_seg_metrics_1y.predict_proba(data_1y['test']['X'])

    preds_seg_5y_train = best_seg_metrics_5y.predict_proba(data_5y['train']['X'])
    preds_seg_5y_test = best_seg_metrics_5y.predict_proba(data_5y['test']['X'])
    
    
    preds_seg_1y_df = pd.concat([pd.DataFrame({'id':[x for x in data_1y['test']['id']], 
                                            'seg_pred':[x for x in preds_seg_1y_test[:,1]]}),
                                pd.DataFrame({'id':[x for x in data_1y['train']['id']], 
                                            'seg_pred':[x for x in preds_seg_1y_train[:,1]]})])
    
    preds_seg_5y_df = pd.concat([pd.DataFrame({'id':[x for x in data_5y['test']['id']], 
                                            'seg_pred':[x for x in preds_seg_5y_test[:,1]]}),
                                pd.DataFrame({'id':[x for x in data_5y['train']['id']], 
                                            'seg_pred':[x for x in preds_seg_5y_train[:,1]]})])
    
    return preds_seg_1y_df, preds_seg_5y_df
def mergePredictions(clin_predictions, dl_predictions, seg_predictions):
    """
    Merge predictions from clinical model, 2D imaging deep learning predictions and LR segmentation metric predictions
    Inputs:
        -clin_predictions: dictionary containing train/test splits with predictions/ids for each
        -dl_predictions: pd.DataFrame containing score_0 and score_1, the unnormalized final model outputs from 2D imaging model
        -seg_predictions: pd.DataFrame containing id and seg_pred, the predictions from segmentation metric for each id
    Outputs:
        -ensembled_predictions: dictionary containing train/test splits with X/y/ids for each where X is clin/img/seg predictions
    """
    #Clinical predictions
    
    #DL Imaging predictions
    dl_predictions['dl_img_class_1_prob'] = dl_predictions.apply(lambda x: softmax(np.array([x['score_0'],x['score_1']]))[1], axis=1)
    
    dl_img_pred_dict = dl_predictions.set_index('anon_id')['dl_img_class_1_prob'].to_dict()
    seg_pred_dict = seg_predictions.set_index('id')['seg_pred'].to_dict()
    ensembled_predictions = {}
    
    for split in ['train','test']:
        ensembled_predictions[split] = {}
        label_dict = dict(zip(clin_predictions[split]['id'], list(clin_predictions[split]['y'])))
        clin_pred_dict = dict(zip(clin_predictions[split]['id'], list(clin_predictions[split]['clin_class_1_preds'])))

        new_data = {k:[clin_pred_dict[k],dl_img_pred_dict[k],seg_pred_dict[k],y] for k,y in label_dict.items()}
        
        new_data_df = pd.DataFrame.from_dict(new_data, orient='index',
                       columns=['clin_pred', 'dl_img_pred','seg_pred','y'])
    
        ensembled_predictions[split]['X'] = new_data_df[['clin_pred', 'dl_img_pred','seg_pred']]
        ensembled_predictions[split]['y'] = new_data_df.y
        ensembled_predictions[split]['id'] = new_data_df.index[0]
        
    
    return ensembled_predictions
def main():
    # 1y model ensemble
    main_data_dir = '/PATH_TO/data/'
    preds_seg_1y_df, preds_seg_5y_df = get_seg_preds()
    clin_preds_1y = load_clin_predictions(main_data_dir, desired_cohort='1y')
    clin_preds_5y = load_clin_predictions(main_data_dir, desired_cohort='5y')
    dl_img_preds_1y = get_dl_preds(desired_cohort='1y')
    dl_img_preds_5y = get_dl_preds(desired_cohort='5y')

    ensemble_data_1y = mergePredictions(clin_preds_1y, dl_img_preds_1y, preds_seg_1y_df)
    ensemble_data_5y = mergePredictions(clin_preds_5y, dl_img_preds_5y, preds_seg_5y_df)
    


    grids, grid_dict = makePipelines()
    best_stacked_1y = {}
    best_stacked_5y = {}
    for i,g in enumerate(grids):
        
        print(f"Stacking with {grid_dict[i]}")
        g.fit(ensemble_data_1y['train']['X'], ensemble_data_1y['train']['y'])
        print(f'\tBest params:{g.best_params_}')
        print(f'\tBest training AUROC: {g.best_score_:.4f}')
        best_stacked_1y[i] = g.best_estimator_

    
        print(f"Stacking with {grid_dict[i]}")
        g.fit(ensemble_data_5y['train']['X'], ensemble_data_5y['train']['y'])
        print(f'\tBest params:{g.best_params_}')
        print(f'\tBest training AUROC: {g.best_score_:.4f}')
        best_stacked_5y[i] = g.best_estimator_

    ensemble_data_1y['train']['img_ehr_pred'] = best_stacked_1y[0].predict_proba(ensemble_data_1y['train']['X'])[:,1]
    ensemble_data_1y['test']['img_ehr_pred'] = best_stacked_1y[0].predict_proba(ensemble_data_1y['test']['X'])[:,1]


    ensemble_data_5y['train']['img_ehr_pred'] = best_stacked_5y[0].predict_proba(ensemble_data_5y['train']['X'])[:,1]
    ensemble_data_5y['test']['img_ehr_pred'] = best_stacked_5y[0].predict_proba(ensemble_data_5y['test']['X'])[:,1]
    _ = showAllMetrics(ensemble_data_1y['test']['y'].values, best_stacked_1y[0].predict_proba(ensemble_data_1y['test']['X'])[:,1])
    _ = showAllMetrics(ensemble_data_5y['test']['y'].values, best_stacked_5y[0].predict_proba(ensemble_data_5y['test']['X'])[:,1])
    joblib.dump(ensemble_data_1y, '/PATH_TO/predictions/img_ehr_seg_pred_1y.pkl')
    joblib.dump(ensemble_data_5y, '/PATH_TO/predictions/img_ehr_seg_pred_5y.pkl')
if __name__=='__main__':
    main()