import pandas as pd
from ml_models_utils import *
import joblib
import sys
sys.path.append('../')
from fusion.IHD_ml_models_segFts_2Dimg_ontology_uncorr_ensemble import get_seg_preds, load_clin_predictions, get_dl_preds, mergePredictions

def convert_df_cols_to_dict(df, id_col, out_col):
    """
    converts two columns from pandas dataframe (df) into dictionary, such that:
        {id_col:out_col}
    """
    return df[[id_col, out_col]].set_index(id_col).to_dict()[out_col]

def get_cohort_labels(data_dir): 
    cohort_data_1yr = pd.read_csv(data_dir+'IHD_8139_1y_train_val_test_split.csv')
    cohort_data_5yr = pd.read_csv(data_dir+'IHD_8139_5y_train_val_test_split.csv')

    return {'1y_labels': convert_df_cols_to_dict(cohort_data_1yr, 'anon_id', '1y_label'), 
            '1y_set':convert_df_cols_to_dict(cohort_data_1yr, 'anon_id', 'set'), 
            '5y_labels':convert_df_cols_to_dict(cohort_data_5yr, 'anon_id', '5y_label'), 
            '5y_set':convert_df_cols_to_dict(cohort_data_5yr, 'anon_id', 'set')}

def get_frs_pce(data_dir):
    baselines_dir = {}
    baselines_dir['frs_1y'] = convert_df_cols_to_dict(pd.read_csv(data_dir+'IHD_8139_1y_framingham_scores.csv'), 'anon_id', 'frs')
    baselines_dir['frs_5y'] = convert_df_cols_to_dict(pd.read_csv(data_dir+'IHD_8139_5y_framingham_scores.csv'), 'anon_id', 'frs')
    baselines_dir['pce_1y'] = convert_df_cols_to_dict(pd.read_csv(data_dir+'IHD_8139_5y_PCE_scores.csv'), 'anon_id', 'pce_risk')
    baselines_dir['pce_5y'] = convert_df_cols_to_dict(pd.read_csv(data_dir+'IHD_8139_5y_PCE_scores.csv'), 'anon_id', 'pce_risk')

    return baselines_dir

def unify_labels_baselines(labels_dict, baselines_dict):
    unified_dict = {'1y':{}, '5y':{}}
    for split in set(labels_dict['1y_set'].values()):
        unified_dict['1y'][split] = {}
        unified_dict['5y'][split] = {}
        #labels
        unified_dict['1y'][split]['label'] = {k:int(labels_dict['1y_labels'][k]) for k,v in labels_dict['1y_set'].items() if v == split}
        unified_dict['5y'][split]['label'] = {k:int(labels_dict['5y_labels'][k]) for k,v in labels_dict['5y_set'].items() if v == split}
        #frs
        unified_dict['1y'][split]['frs'] = {k:baselines_dict['frs_1y'][k] for k in unified_dict['1y'][split]['label'].keys()}
        unified_dict['5y'][split]['frs'] = {k:baselines_dict['frs_5y'][k] for k in unified_dict['5y'][split]['label'].keys()}
        #pce
        unified_dict['1y'][split]['pce'] = {k:baselines_dict['pce_1y'][k] for k in unified_dict['1y'][split]['label'].keys()}
        unified_dict['5y'][split]['pce'] = {k:baselines_dict['pce_5y'][k] for k in unified_dict['5y'][split]['label'].keys()}
        
    return unified_dict
def flatten_train_test_data_dict(d, field1, field2, index=None):
    """
    flattens custom dict split into train/test into a single 1
    """
    if index is None:
        return {**{x[0]:x[1] for x in zip(list(d['train'][field1]), list(d['train'][field2]))},
                **{x[0]:x[1] for x in zip(list(d['test'][field1]), list(d['test'][field2]))}}
    else:
        return {**{x[0]:x[1] for x in zip(list(d['train'][field1].index), list(d['train'][field2]))},
                **{x[0]:x[1] for x in zip(list(d['test'][field1].index), list(d['test'][field2]))}}

def combine_all_preds(baselines_dict, seg_dict, imaging_dict, ehr_dict, pce_seg_dict, img_ehr_dict, img_ehr_seg_dict):
    all_preds_dict = {}
    for split in ['train','val','test']:
        for ID in baselines_dict[split]['label'].keys():
            all_preds_dict[ID] = [split,
                                 baselines_dict[split]['label'][ID], 
                                 baselines_dict[split]['frs'][ID], 
                                 baselines_dict[split]['pce'][ID],
                                 seg_dict[ID],
                                 imaging_dict[ID],
                                 ehr_dict[ID],
                                 pce_seg_dict[ID],
                                 img_ehr_dict[ID],
                                 img_ehr_seg_dict[ID]]
    
    all_preds_df = pd.DataFrame.from_dict(all_preds_dict, orient='index',
                       columns=['set', 'label', 'frs', 'pce_risk', 'seg_risk','img_pred','clin_pred','pce_seg_model_pred','img_clin_fusion_preds','img_clin_seg_fusion_preds'])
    
    all_preds_df['anon_id'] = all_preds_df.index
    return all_preds_df.reset_index(drop=True)

def apply_softmax_and_return_class1_prob(df, id_col='anon_id', score_cols=['score_0','score_1']):
    df['class_1_prob'] = df.apply(lambda x: softmax(np.array([x[col] for col in score_cols]))[1], axis=1)
    return convert_df_cols_to_dict(df, 'anon_id', 'class_1_prob')

def main():
    cohort_data_dir = '/PATH_TO/data/'
    predictions_dir ='/PATH_TO/predictions/'
    #labels
    labels = get_cohort_labels(cohort_data_dir)

    #baselines
    frs_pce = get_frs_pce(cohort_data_dir)
    
    labels_baselines = unify_labels_baselines(labels,frs_pce)
    
    #segmentation only
    preds_seg_1y_df, preds_seg_5y_df = get_seg_preds()
    preds_seg_1y = convert_df_cols_to_dict(preds_seg_1y_df, 'id', 'seg_pred')
    preds_seg_5y = convert_df_cols_to_dict(preds_seg_5y_df, 'id', 'seg_pred')

    #clinical only
    clin_preds_1y = load_clin_predictions(cohort_data_dir, desired_cohort='1y')
    clin_preds_1y = flatten_train_test_data_dict(clin_preds_1y, 'id','clin_class_1_preds')
    clin_preds_5y = load_clin_predictions(cohort_data_dir, desired_cohort='5y')
    clin_preds_5y = flatten_train_test_data_dict(clin_preds_5y, 'id','clin_class_1_preds')

    #image only
    dl_img_preds_1y = apply_softmax_and_return_class1_prob(get_dl_preds(desired_cohort='1y'))
    dl_img_preds_5y = apply_softmax_and_return_class1_prob(get_dl_preds(desired_cohort='5y'))

    #pce + seg
    ps_ensemble_data_1y = joblib.load(predictions_dir+'img_pce_seg_pred_1y.pkl')
    ps_ensemble_data_5y = joblib.load(predictions_dir+'img_pce_seg_pred_5y.pkl')
    ps_ensemble_1y = flatten_train_test_data_dict(ps_ensemble_data_1y, 'id', 'pce_seg_preds')
    ps_ensemble_5y = flatten_train_test_data_dict(ps_ensemble_data_5y, 'id', 'pce_seg_preds')
    
    #image + clinical
    ic_ensemble_data_1y = joblib.load(predictions_dir+'img_ehr_pred_1y.pkl')
    ic_ensemble_data_5y = joblib.load(predictions_dir+'img_ehr_pred_5y.pkl')
    ic_ensemble_1y = flatten_train_test_data_dict(ic_ensemble_data_1y, 'X', 'img_ehr_pred', index=True)
    ic_ensemble_5y = flatten_train_test_data_dict(ic_ensemble_data_5y, 'X', 'img_ehr_pred', index=True)
    
    #image + clinical + segmentation
    ics_ensemble_data_1y = joblib.load(predictions_dir+'img_ehr_seg_pred_1y.pkl')
    ics_ensemble_data_5y = joblib.load(predictions_dir+'img_ehr_seg_pred_5y.pkl')
    ics_ensemble_1y = flatten_train_test_data_dict(ics_ensemble_data_1y, 'X', 'img_ehr_pred', index=True)
    ics_ensemble_5y = flatten_train_test_data_dict(ics_ensemble_data_5y, 'X', 'img_ehr_pred', index=True)

    #unify into single dataframe
    
    unified_preds_1y = combine_all_preds(baselines_dict=labels_baselines['1y'], 
                                        seg_dict=preds_seg_1y, 
                                        imaging_dict=dl_img_preds_1y, 
                                        ehr_dict=clin_preds_1y, 
                                        pce_seg_dict=ps_ensemble_1y, 
                                        img_ehr_dict=ic_ensemble_1y, 
                                        img_ehr_seg_dict=ics_ensemble_1y)
    unified_preds_5y = combine_all_preds(baselines_dict=labels_baselines['5y'], 
                                        seg_dict=preds_seg_5y, 
                                        imaging_dict=dl_img_preds_5y, 
                                        ehr_dict=clin_preds_5y, 
                                        pce_seg_dict=ps_ensemble_5y, 
                                        img_ehr_dict=ic_ensemble_5y, 
                                        img_ehr_seg_dict=ics_ensemble_5y)
    
    #save 
    unified_preds_1y.to_csv(predictions_dir+'IHD_8139_preds_all_1y.csv', index=False)
    unified_preds_5y.to_csv(predictions_dir+'IHD_8139_preds_all_5y.csv', index=False)
if __name__=='__main__':
    main()
