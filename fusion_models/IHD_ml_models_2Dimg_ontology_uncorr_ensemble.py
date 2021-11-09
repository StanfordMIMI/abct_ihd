from ml_models_utils import *
import joblib
import sys

def get_model_cols(fpath):
    with open(fpath,'r') as fin:
        cols = [x.strip() for x in fin.readlines()]
    return cols

def load_data(section_path, cci_path, col_data_path, verbose=True):
    desired_cols = get_model_cols(col_data_path)
    section_data = pd.read_csv(section_path)
    
    
    section_data['smoker'] = section_data['smoker'].astype(bool)
    section_data_train = section_data[(section_data['set']=='train')|(section_data['set']=='val')].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)
    
    
    train_x = section_data_train[desired_cols]
    train_y = section_data_train.label
    train_ids = section_data_train.anon_id

    test_x = section_data_test[desired_cols]
    test_y = section_data_test.label
    test_ids = section_data_test.anon_id
    if verbose:
        print(f'{section_data.shape[0]} patients and {section_data.shape[1]-2} features in cohort')
        print(f"Train set n: {section_data_train.shape[0]}, test set n: {section_data_test.shape[0]}")
        print(f'{train_x.shape[0]+test_x.shape[0]} CTs and {train_x.shape[1]} features in cohort')
        print(f"Percent positive label in train/test set: {100*sum(train_y.values==1)/len(train_y):.1f} % /{100*sum(test_y.values==1)/len(test_y):.1f} %")
    return {'train': {'X':train_x, 'y':train_y, 'id':train_ids}, 'test': {'X':test_x, 'y':test_y, 'id':test_ids}}

def load_clin_predictions(main_data_dir, desired_cohort=''):
    #load data
    model_dirpath = '/PATH_TO/models/'
    if desired_cohort=='1y':
        data_path = main_data_dir + 'IHD_8139_1y_fts.csv'
        col_data_path = model_dirpath+'final_fts_1y_model.csv'
        trained_clf_path = model_dirpath+'best_ontology_fts_xgboost_1y.pkl'
    elif desired_cohort=='5y':
        data_path = main_data_dir + 'IHD_8139_5y_fts.csv'
        col_data_path = model_dirpath+'final_fts_5y_model.csv'
        trained_clf_path = model_dirpath+'best_ontology_fts_xgboost_5y.pkl'
    else:
        raise ValueError('Incorrect cohort specified (must be 1y or 5y)')
    data = load_data(data_path, cci_path, col_data_path, verbose=False)

    #load classifier
    clin_clf = joblib.load(trained_clf_path)
    
    #get predictions from clin classifier

    data['train']['clin_class_1_preds'] = clin_clf.predict_proba(data['train']['X'])[:,1]
    data['test']['clin_class_1_preds'] = clin_clf.predict_proba(data['test']['X'])[:,1]

    return data

def get_dl_preds(desired_cohort):
    """"""
    if desired_cohort=='1y':
        fpath = '/PATH_TO/predictions/img_only_preds_1y_efficientnet_1y_1_0.csv' 
    elif desired_cohort=='5y':
        fpath = '/PATH_TO/predictions/img_only_preds_5y_efficientnet_5y_1_0.csv'
    else: 
        raise ValueError('Incorrect cohort specified (must be 1y or 5y)')
    return pd.read_csv(fpath, sep='\t')[['anon_id','score_0','score_1']]

def mergePredictions(clin_predictions, dl_predictions):
    """
    Merge predictions from clinical models as well as deep learning predictions
    Inputs:
        -clin_predictions: dictionary containing train/test splits with predictions/ids for each
        -dl_predictions: pd.DataFrame containing score_0 and score_1, the unnormalized final model outputs from 2D imaging model
    Outputs:
        -ensembled_predictions: dictionary containing train/test splits with X/y/ids for each where X is clin/dl predictions
    """
    #Clinical predictions
    
    #DL Imaging predictions
    dl_predictions['dl_img_class_1_prob'] = dl_predictions.apply(lambda x: softmax(np.array([x['score_0'],x['score_1']]))[1], axis=1)
    
    dl_img_pred_dict = dl_predictions.set_index('anon_id')['dl_img_class_1_prob'].to_dict()
    
    ensembled_predictions = {}
    
    for split in ['train','test']:
        ensembled_predictions[split] = {}
        label_dict = dict(zip(clin_predictions[split]['id'], list(clin_predictions[split]['y'])))
        clin_pred_dict = dict(zip(clin_predictions[split]['id'], list(clin_predictions[split]['clin_class_1_preds'])))
        
        new_data = {k:[clin_pred_dict[k],dl_img_pred_dict[k],y] for k,y in label_dict.items()}
        
        new_data_df = pd.DataFrame.from_dict(new_data, orient='index',
                       columns=['clin_pred', 'dl_img_pred','y'])
    
        ensembled_predictions[split]['X'] = new_data_df[['clin_pred', 'dl_img_pred']]
        ensembled_predictions[split]['y'] = new_data_df.y
        ensembled_predictions[split]['id'] = new_data_df.index[0]
        
    
    return ensembled_predictions

def makePipelines(scoring='AUC'):
    """
    Makes pipelines for logistic regression
    """
    pipe_lr = Pipeline([('clf',LogisticRegression(random_state=17, max_iter=3000))])
    
    grid_params_lr = [{'clf__C': [.001, 0.1, 1, 10, 100], 'clf__solver': ['liblinear']}]
    if scoring =='AUC':
        scoring = {'AUC':'roc_auc'}
        refit='AUC'
    elif scoring == 'Sensitivity':
        scoring = {'sensitivity':'recall'}
        refit='sensitivity'   
    gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=grid_params_lr, scoring=scoring,cv=10, refit=refit, verbose=1, n_jobs=-1)

    grids = [gs_lr]
    grid_dict = {0:'LR'}

    return grids, grid_dict

def main():
    # 1y model ensemble
    main_data_dir = '/PATH_TO/data/'
    clin_preds_1y = load_clin_predictions(main_data_dir, desired_cohort='1y')
    clin_preds_5y = load_clin_predictions(main_data_dir, desired_cohort='5y')
    dl_img_preds_1y = get_dl_preds(desired_cohort='1y')
    dl_img_preds_5y = get_dl_preds(desired_cohort='5y')

    ensemble_data_1y = mergePredictions(clin_preds_1y, dl_img_preds_1y)
    ensemble_data_5y = mergePredictions(clin_preds_5y, dl_img_preds_5y)


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

    #show all metrics for LR stacking
    print("Stacking results with Logistic regression on test set:")
    _ = showAllMetrics(ensemble_data_1y['test']['y'].values, best_stacked_1y[0].predict_proba(ensemble_data_1y['test']['X'])[:,1])
    _ = showAllMetrics(ensemble_data_5y['test']['y'].values, best_stacked_5y[0].predict_proba(ensemble_data_5y['test']['X'])[:,1])

    ensemble_data_1y['train']['img_ehr_pred'] = best_stacked_1y[0].predict_proba(ensemble_data_1y['train']['X'])[:,1]
    ensemble_data_1y['test']['img_ehr_pred'] = best_stacked_1y[0].predict_proba(ensemble_data_1y['test']['X'])[:,1]


    ensemble_data_5y['train']['img_ehr_pred'] = best_stacked_5y[0].predict_proba(ensemble_data_5y['train']['X'])[:,1]
    ensemble_data_5y['test']['img_ehr_pred'] = best_stacked_5y[0].predict_proba(ensemble_data_5y['test']['X'])[:,1]

    joblib.dump(ensemble_data_1y, '/PATH_TO/predictions/img_ehr_pred_1y.pkl')
    joblib.dump(ensemble_data_5y, '/PATH_TO/predictions/img_ehr_pred_5y.pkl')
if __name__=='__main__':
    main()