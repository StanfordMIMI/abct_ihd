import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

def load_data(section_path, label_mapper={'Control':0, 'Ischaemic heart diseases':1}):
    computed_l3_metrics = pd.read_csv('/PATH_TO/data/seg_fts.csv').drop(['id'], axis=1) #file containing BC metrics for all individuals
    section_data = pd.read_csv(section_path)[['id','label','set']]
    section_data = pd.merge(section_data, computed_l3_metrics, how='left', left_on='id', right_on='id').drop('id', axis=1)
    
    section_data_train = section_data[section_data['set']=='train'].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)

    train_x = section_data_train.drop(columns='label')
    train_y = section_data_train.label.map(label_mapper)
    test_x = section_data_test.drop(columns='label')
    test_y = section_data_test.label.map(label_mapper)
    
    return {'train': {'X':train_x, 'y':train_y}, 'test': {'X':test_x, 'y':test_y}}

def makePipelines(scoring='AUC', univariate=True):
    """
    Makes pipelines for logistic regression
    """
    pipe_lr = Pipeline([('clf',LogisticRegression(random_state=17, max_iter=30000,multi_class='auto'))])

    if univariate: 
        grid_params_lr = [{'clf__penalty': ['none'], 'clf__solver': ['lbfgs']}]
    else:
        grid_params_lr = [{'clf__C': [10, 5, 1, 0.5, 0.1, 0.001], 'clf__solver': ['lbfgs']}]
    
    scoring = {'AUC':'roc_auc'}
    refit='AUC'

    gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=grid_params_lr, scoring=scoring,cv=10, refit=refit)
    
    grids = [gs_lr]
    grid_dict = {0:'Logistic regression'}
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

def getBestClassifiers(data_dict, cols_to_use, grids, grid_dict, metric):

    train_x, test_x = data_dict['train'][cols_to_use], data_dict['test'][cols_to_use]
    train_y, test_y = data_dict['train']['y'], data_dict['test']['y']
    
    #Get best classifier
    best_metric_clf = [None]*len(grids)
    for idx, gs in enumerate(grids):
        best_metric_clf[idx] = getBestClassifier(gs, grid_dict[idx], train_x, test_x, \
                                                  train_y, test_y, metric)
    return best_metric_clf, train_x, test_x, train_y, test_y

def prepareDataDict(data_dict):

    data_dict['train']['muscle_fat'] = data_dict['train']['X'][['muscle_HU', 'vat_sat_ratio']]
    data_dict['test']['muscle_fat'] = data_dict['test']['X'][['muscle_HU', 'vat_sat_ratio']]

    return data_dict


def main():
    
    #Prepare data
    data_path_1y = '/PATH_TO/data/1y_cohort.csv'
    data_1y = load_data(data_path_1y, label_mapper={-2:0, -1:0, 0:0, 1:1, 2:0, 3:0})
    data_path_5y = '/PATH_TO/data/5y_cohort.csv'
    data_5y = load_data(data_path_5y, label_mapper={-2:0, -1:0, 0:0, 1:1, 2:1, 3:0})
    
    data_1y = prepareDataDict(data_1y)
    data_5y = prepareDataDict(data_5y)
    
    #Prepare model
    grids, grid_dict = makePipelines(univariate=False)

    #Train and save: 1y cohort
    best_auc_clfs1y, _, _, _, _ = getBestClassifiers(data_1y, 'muscle_fat', grids, grid_dict, 'AUC')

    save_folder = '/PATH_TO/models/best_segmentation_fts_'

    for i, clf1y in enumerate(best_auc_clfs1y):
        save_fpath = save_folder + '1y_'.join(grid_dict[i].split()) + '.pkl'
        joblib.dump(clf1y, save_fpath)

    #Train and save: 1y cohort
    best_auc_clfs5y, _, _, _, _ = getBestClassifiers(data_5y, 'muscle_fat', grids, grid_dict, 'AUC')
    
    for i, clf5y in enumerate(best_auc_clfs5y):
        save_fpath = save_folder + '5y_'.join(grid_dict[i].split()) + '.pkl'
        joblib.dump(clf5y, save_fpath)

if __name__ =='__main__':
    main()
