from ml_models_utils import *
from copy import deepcopy
import joblib
import sys 

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

def get_corr_fts(df, threshold):
    corr_cols = []
    corr_matrix = df.corr()
    for i in range(len(df.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                corr_cols.append(corr_matrix.columns[i])
    return set(corr_cols)

def remove_corr_cols(X_train, X_test, threshold):
    to_remove = get_corr_fts(X_train, threshold)
    return X_train.drop(labels=to_remove, axis=1, inplace=False), X_test.drop(labels=to_remove, axis=1, inplace=False)

def customScaler(X_train, X_test, standard_columns, minMax_columns):
    """
    Rescales train/test data by performing standard rescaling on standard_columns
    and minMax scaling on minMax_columns
    """
    standard = StandardScaler()
    minmax = MinMaxScaler()

    standard_rescale_train = X_train[standard_columns]
    standard_rescale_test = X_test[standard_columns]
    standard.fit(standard_rescale_train)
    
    minmax_rescale_train = X_train[minMax_columns]
    minmax_rescale_test = X_test[minMax_columns]
    minmax.fit(minmax_rescale_train)

    X_train_rescaled = pd.concat([pd.DataFrame(standard.transform(standard_rescale_train), columns=standard_columns), 
                                  pd.DataFrame(minmax.transform(minmax_rescale_train), columns=minMax_columns)], 
                                    axis=1)
    X_test_rescaled = pd.concat([pd.DataFrame(standard.transform(standard_rescale_test), columns=standard_columns), 
                                 pd.DataFrame(minmax.transform(minmax_rescale_test), columns=minMax_columns)],
                                    axis=1)
    return X_train_rescaled, X_test_rescaled

def makePipelines(scoring='AUC'):
    """
    Makes pipelines for XGBoost
    """
    pipe_xgb = Pipeline([('clf', XGBClassifier(seed=17))])
    
    param_range_fl = [10, 5, 1.0, 0.5, 0.1]
    param_range_mx_dpth = [1,2,3,4,5,6]
    param_range_colsample_bytree = [.2,.3,.4,.5,.6,.7,.8,.9,1.]
    param_range_min_child_weight = [0.1,0.5,1]
    param_range_subsample = [.6,.7,.8,.9,1.]
    param_range_lr = [.3, .1, .05, .01]
    
    
    grd_params_xgb = [{'clf__colsample_bytree':param_range_colsample_bytree, 'clf__gamma': param_range_fl, 'clf__max_depth': param_range_mx_dpth,
                       'clf__min_child_weight': param_range_min_child_weight, 'clf__subsample': param_range_subsample, 'clf__learning_rate':param_range_lr}]
    
    if scoring =='AUC':
        scoring = {'AUC':'roc_auc'}
        refit='AUC'
    elif scoring == 'Sensitivity':
        scoring = {'sensitivity':'recall'}
        refit='sensitivity'
        
    gs_xgb = GridSearchCV(estimator=pipe_xgb, param_grid=grd_params_xgb, scoring=scoring,cv=10, refit=refit, verbose=2, n_jobs=-1)
    
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
    #Get best classifier
    best_metric_clf = [None]*len(grids)
    for idx, gs in enumerate(grids):
        best_metric_clf[idx] = getBestClassifier(gs, grid_dict[idx], X_train_r, X_test_r, \
                                                  train_y, test_y, metric)
    return best_metric_clf, {'train':{'X':train_x, 'X_r': X_train_r, 'y':train_y}, 'test':{'X':test_x, 'X_r': X_test_r, 'y':test_y}}

def get_corr_data_subsets(orig_data, corr_thresholds):
    data_corrs = {}
    for corr_thresh in corr_thresholds:
        data_copy = deepcopy(orig_data)
        data_copy['train']['X'], data_copy['test']['X'] = remove_corr_cols(data_copy['train']['X'], data_copy['test']['X'], corr_thresh)
        data_corrs[str(corr_thresh)] = data_copy
    return data_corrs

def main():
    main_data_dir = '/PATH_TO/data/'
    #Set up 
    grids, grid_dict = makePipelines()
    corr_grid = [np.Inf, .9,.8,.7,.6,.5,.4,.3,.2,.1]
    #1y
    print('-'*50,'    1y cohort    ','-'*50)
    data_path_1y_noimp = main_data_dir+'IHD_8139_1y_fts.csv'
    data_1y_noimp = load_data(data_path_1y_noimp)

    data_1y_noimp_corrs = get_corr_data_subsets(data_1y_noimp,corr_grid)

    for k in data_1y_noimp_corrs.keys():
        print(f'Running experiments with correlation threshold {k} for 1y cohort')
        best_auc_clfs_1y_noimp, data_1y_noimp_r = splitAndGetBestClassifiers(data_1y_noimp_corrs[k], grids, grid_dict, 'AUC', rescale=False)
    
    #5y
    print('-'*50,'    5y cohort    ','-'*50)
    data_path_5y_noimp = main_data_dir + 'IHD_8139_5y_fts.csv'
    data_5y_noimp = load_data(data_path_5y_noimp)
    
    data_5y_noimp_corrs = get_corr_data_subsets(data_5y_noimp,corr_grid)
    
    for k in data_5y_noimp_corrs.keys():
        print(f'Running experiments with correlation threshold {k} for 5y cohort')
        best_auc_clfs_5y_noimp, data_5y_noimp_r = splitAndGetBestClassifiers(data_5y_noimp_corrs[k], grids, grid_dict, 'AUC', rescale=False)

if __name__=='__main__':
    sys.stdout=open('../logs/'+'IHD_8139_clin_only_elim_corr_exploration.log',"w")
    main()
    sys.stdout.close()