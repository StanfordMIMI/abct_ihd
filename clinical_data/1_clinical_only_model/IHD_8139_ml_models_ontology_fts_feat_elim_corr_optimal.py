from IHD_8139_ml_models_ontology_fts_feat_elim_corr_exploration import *


def load_data(fpath):
    
    section_data = pd.read_csv(fpath)
    
    print(f'{section_data.shape[0]} patients and {section_data.shape[1]-2} features in cohort')
    section_data['smoker'] = section_data['smoker'].astype(bool)
    section_data_train = section_data[(section_data['set']=='train')|(section_data['set']=='val')].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)
    print(f"Train set n: {section_data_train.shape[0]}, test set n: {section_data_test.shape[0]}")

    train_x = section_data_train.drop(columns='label')
    train_y = section_data_train.label
    test_x = section_data_test.drop(columns='label')
    test_y = section_data_test.label
    
    print(f"Percent positive label in train/test set: {100*sum(train_y.values==1)/len(train_y):.1f} % /{100*sum(test_y.values==1)/len(test_y):.1f} %")
    return {'train': {'X':train_x, 'y':train_y}, 'test': {'X':test_x, 'y':test_y}}

def getTrainedClassifier(data, params, rescale=True):
    train_x, train_y = data['train']['X'], data['train']['y']
    test_x, test_y = data['test']['X'], data['test']['y']
    if rescale:
        #Rescale data
        standard_columns = ['latest_value_diastolic','latest_value_systolic', 
                           'exp_wt_lab_diastolic',
                            'exp_wt_lab_systolic', 'bmi', 'age',
                           'exp_wt_lab_chol_hdl', 'exp_wt_lab_chol_ldl', 'exp_wt_lab_chol_total', 'exp_wt_lab_gluc',
                           'exp_wt_lab_hba1c', 'exp_wt_lab_trig', 'latest_value_chol_hdl', 'latest_value_chol_ldl', 
                            'latest_value_chol_total', 'latest_value_gluc', 'latest_value_hba1c', 'latest_value_trig']

        minMaxColumns = [x for x in train_x.columns if x not in standard_columns]

        X_train_r, X_test_r = customScaler(train_x, test_x, standard_columns, minMaxColumns)
    else:
        X_train_r, X_test_r = train_x, test_x
    #Get trained classifier
    best_metric_clf = XGBClassifier(seed=17, **params)
    print(best_metric_clf)
    best_metric_clf.fit(X_train_r, train_y)
    return best_metric_clf, {'train':{'X':train_x, 'X_r': X_train_r, 'y':train_y}, 'test':{'X':test_x, 'X_r': X_test_r, 'y':test_y}}

def showTrainTestPerformance(trained_clf, data):
    print(f"Training Performance")
    y_scores_train = trained_clf.predict_proba(data['train']['X_r'])
    _,_,_,_,thresh = showAllMetrics(data['train']['y'].values, y_scores_train[:,1])
    print()
    print(f"Test Performance")
    y_scores_test = trained_clf.predict_proba(data['test']['X_r'])
    showAllMetrics(data['test']['y'].values, y_scores_test[:,1], optimal_threshold=thresh)

def main():
    main_data_dir = '/PATH_TO/data/'
    save_folder = '/PATH_TO/models/'
    # 1y outcome
    print('-'*50,'    1y cohort    ','-'*50)
    data_path_1y_noimp = main_data_dir+'IHD_8139_1y_fts.csv'
    
    data_1y_noimp = load_data(data_path_1y_noimp)
    
    data_1y_noimp_corrs = {}

    for corr_thresh in [.5]:
        data_1y_copy = deepcopy(data_1y_noimp)
        data_1y_copy['train']['X'], data_1y_copy['test']['X'] = remove_corr_cols(data_1y_copy['train']['X'], data_1y_copy['test']['X'], corr_thresh)
        data_1y_noimp_corrs[str(corr_thresh)] = data_1y_copy

    
    best_params={'colsample_bytree': .9, 'gamma': 5, 'max_depth': 5, 'min_child_weight': 1, 'subsample': .9, 'learning_rate':.05, 'early_stopping_rounds':3}
    
    
    best_auc_clfs_1y, data_1y_r = getTrainedClassifier(data_1y_noimp_corrs[str(.5)], best_params, rescale=False)

    showTrainTestPerformance(best_auc_clfs_1y, data_1y_r)
    
    
    joblib.dump(best_auc_clfs_1y, save_folder+'best_ontology_fts_xgboost'+'_1y.pkl')
    with open(save_folder+'final_fts_1y_model.csv','w') as fout:
        for x in data_1y_r['train']['X'].columns:
            fout.write(x+'\n')
    
    #5y outcome
    print()
    print('-'*50,'    5y cohort    ','-'*50)
    data_path_5y_noimp = main_data_dir + 'IHD_8139_5y_ft_matrix_ontology.csv'
    data_5y_noimp = load_data(data_path_5y_noimp)

    data_5y_noimp_corrs = {}

    for corr_thresh in [.5]:
        data_5y_copy = deepcopy(data_5y_noimp)
        data_5y_copy['train']['X'], data_5y_copy['test']['X'] = remove_corr_cols(data_5y_copy['train']['X'], data_5y_copy['test']['X'], corr_thresh)
        data_5y_noimp_corrs[str(corr_thresh)] = data_5y_copy
    
    best_params={'colsample_bytree': 1, 'gamma': 10, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 1, 'learning_rate':.1, 'early_stopping_rounds':3}
    best_auc_clfs_5y, data_5y_r = getTrainedClassifier(data_5y_noimp_corrs[str(.5)], best_params, rescale=False)

    showTrainTestPerformance(best_auc_clfs_5y, data_5y_r)

    joblib.dump(best_auc_clfs_5y, save_folder+'best_ontology_fts_xgboost'+'_5y.pkl')
    with open(save_folder+'final_fts_5y_model.csv','w') as fout:
        for x in data_5y_r['train']['X'].columns:
            fout.write(x+'\n')   

if __name__=='__main__':
    sys.stdout=open('../logs/'+'IHD_8139_clin_only_elim_corr_optimal.log',"w")
    main()
    sys.stdout.close()
    