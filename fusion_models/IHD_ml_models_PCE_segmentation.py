from ml_models_utils import *
import joblib
import sys

def load_data(section_path, label_col=''):

    computed_l3_metrics = pd.read_csv('/PATH_TO/data/IHD_8139_segmentation_fts.csv')[['anon_id','muscle_HU', 'vat_sat_ratio']] #file containing BC metrics for all individuals
    pce_metrics = pd.read_csv('/PATH_TO/data/IHD_8139_1y_PCE_covariates.csv') #file containing PCE covariates for all individuals
    section_data = pd.read_csv(section_path)[['anon_id',label_col,'set']]
    section_data = section_data.rename({label_col:'label'}, axis='columns')
    section_data = pd.merge(section_data, computed_l3_metrics, how='left', left_on='anon_id', right_on='anon_id').merge(pce_metrics, \
                    how='left', left_on='anon_id', right_on='anon_id')
    section_data['smoker'] = section_data['smoker'].astype(bool)
    section_data['race4calc'] = section_data[section_data['race4calc']=='Black']
    section_data['race4calc'] = section_data['smoker'].astype(bool)

    section_data_train = section_data[(section_data['set']=='train')|(section_data['set']=='val')].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)

    train_ids = section_data_train.anon_id.values
    train_x = section_data_train.drop(columns=['anon_id','label'])
    train_y = section_data_train.label

    test_ids = section_data_test.anon_id.values
    test_x = section_data_test.drop(columns=['anon_id','label'])
    test_y = section_data_test.label
    print(f'{section_data.shape[0]} patients and {section_data.shape[1]-3} features in cohort')
    print(f"Train set n: {train_x.shape[0]}, test set n: {test_x.shape[0]}")
    print(f"Percent positive label in train/test set: {100*sum(train_y.values==1)/len(train_y):.1f} % /{100*sum(test_y.values==1)/len(test_y):.1f} %")
    
    return {'train': {'X':train_x, 'y':train_y, 'id':train_ids}, 'test': {'X':test_x, 'y':test_y, 'id':test_ids}}

def makePipelines(scoring='AUC'):
    """
    Makes pipelines for XGBoost
    """
    pipe_xgb = Pipeline([('clf', XGBClassifier(seed=17))])
    
    
    param_range_colsample_bytree = [.2,.4,.6,.8,1.]
    param_range_fl = [10, 1.0, 0.5, 0.1]
    param_range_mx_dpth = [1,2,3,4,5,6]
    param_range_min_child_weight = [0.1,0.5,1]
    param_range_subsample = [0.5, 0.75, 1]
    
    grd_params_xgb = [{'clf__colsample_bytree':param_range_colsample_bytree, 'clf__gamma': param_range_fl, 'clf__max_depth': param_range_mx_dpth, 
                       'clf__min_child_weight': param_range_min_child_weight, 'clf__subsample': param_range_subsample}]
    
    if scoring =='AUC':
        scoring = {'AUC':'roc_auc'}
        refit='AUC'
    elif scoring == 'Sensitivity':
        scoring = {'sensitivity':'recall'}
        refit='sensitivity'

    gs_xgb = GridSearchCV(estimator=pipe_xgb, param_grid=grd_params_xgb, scoring=scoring,cv=10, refit=refit, verbose=1, n_jobs=-1)
    
    grids = [gs_xgb]
    grid_dict = {0:'XGBoost'}

    return grids, grid_dict

def getBestClassifier(gridSearch, clf_name, X_train, X_test, y_train, y_test, metric='AUC'):
    """
    Explore hyperparameters and return best classifier according to prespecified metric
    """
    print(f'\nModel: {clf_name}')
    gridSearch.fit(X_train, y_train)
    print(f'Best params:{gridSearch.best_params_}')
    print(f'Best training {metric}: {gridSearch.best_score_}')
    
    best_classifier = gridSearch.best_estimator_
    
    return best_classifier

def splitAndGetBestClassifiers(data, grids, grid_dict, metric, rescale=True):
    train_x, train_y = data['train']['X'], data['train']['y']
    test_x, test_y = data['test']['X'], data['test']['y']
    #Get best classifier
    best_metric_clf = [None]*len(grids)
    for idx, gs in enumerate(grids):
        best_metric_clf[idx] = getBestClassifier(gs, grid_dict[idx], train_x, test_x, \
                                                  train_y, test_y, metric)
    return best_metric_clf

def showTrainTestPerformance(trained_clf, data):
    print(f"Training Performance")
    y_scores_train = trained_clf.predict_proba(data['train']['X'])
    _ = showAllMetrics(data['train']['y'].values, y_scores_train[:,1])
    print()
    print(f"Test Performance")
    y_scores_test = trained_clf.predict_proba(data['test']['X'])
    showAllMetrics(data['test']['y'].values, y_scores_test[:,1])

def main():
    data_path_1y = '/PATH_TO/data/IHD_8139_1y_train_val_test_split.csv'
    data_1y = load_data(data_path_1y, '1y_label')
    data_path_5y = '/PATH_TO/data/IHD_8139_5y_train_val_test_split.csv'
    data_5y = load_data(data_path_5y, '5y_label')

    grids, grid_dict = makePipelines()

    best_auc_clfs_1y = splitAndGetBestClassifiers(data_1y, grids, grid_dict, 'AUC', rescale=False)
    showTrainTestPerformance(best_auc_clfs_1y[0], data_1y)

    data_1y['train']['pce_seg_preds'] = best_auc_clfs_1y[0].predict_proba(data_1y['train']['X'])[:,1]
    data_1y['test']['pce_seg_preds'] = best_auc_clfs_1y[0].predict_proba(data_1y['test']['X'])[:,1]

    best_auc_clfs_5y = splitAndGetBestClassifiers(data_5y, grids, grid_dict, 'AUC', rescale=False)
    showTrainTestPerformance(best_auc_clfs_5y[0], data_5y)

    data_5y['train']['pce_seg_preds'] = best_auc_clfs_5y[0].predict_proba(data_5y['train']['X'])[:,1]
    data_5y['test']['pce_seg_preds'] = best_auc_clfs_5y[0].predict_proba(data_5y['test']['X'])[:,1]

    joblib.dump(data_1y, '/PATH_TO/predictions/img_pce_seg_pred_1y.pkl')
    joblib.dump(data_5y,'/PATH_TO/predictions/img_pce_seg_pred_5y.pkl')
if __name__=='__main__':
    sys.stdout=open('../logs/'+'IHD_8139_seg_PCE_fusion.log',"w")
    main()
    sys.stdout.close()