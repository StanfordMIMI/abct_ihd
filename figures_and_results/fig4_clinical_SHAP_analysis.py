import shap
import pandas as pd
import joblib
from collections import Counter
from textwrap import fill
import re 
import matplotlib.font_manager
from shap.plots.colors import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt


DATA_DIR = '/PATH_TO/data/'
MODEL_DIR = '/PATH_TO/models'

def load_data(fpath, col_data_path):

    desired_cols = get_cols(col_data_path)
    section_data = pd.read_csv(fpath)
    
    
    section_data['smoker'] = section_data['smoker'].astype(bool)
    section_data_train = section_data[(section_data['set']=='train')|(section_data['set']=='val')].drop('set', axis=1)
    section_data_test = section_data[section_data['set']=='test'].drop('set', axis=1)
    

    train_x = section_data_train.drop(columns='label')[desired_cols]
    train_y = section_data_train.label
    train_id2idx = {v:k for k,v in dict(section_data_train['anon_id']).items()}
    train_idx2newidx = {v:k for k,v in dict(section_data_train.reset_index(level=0, inplace=False)['index']).items()}
    test_x = section_data_test.drop(columns='label')[desired_cols]
    test_y = section_data_test.label
    test_id2idx = {v:k for k,v in dict(section_data_test['anon_id']).items()}
    test_idx2newidx = {v:k for k,v in dict(section_data_test.reset_index(level=0, inplace=False)['index']).items()}

    return {'train': {'X':train_x, 'y':train_y, 'id2idx':train_id2idx, 'idx2newidx':train_idx2newidx}, 
            'test': {'X':test_x, 'y':test_y, 'id2idx':test_id2idx, 'idx2newidx':test_idx2newidx}}

def get_cols(fpath):
    with open(fpath,'r') as fin:
        cols = [x.strip() for x in fin.readlines()]
    return cols

def round_float(s):
    '''1. if s is float, round it to 0 decimals
       2. else return s as is
    '''
    s = float(s)
    r = round(s, 1)
    if str(float(r)).endswith("0"):
        r=int(r)
    return str(r)

def make_force_plot(ID, split, data, explainer, shap_vals, code2name, save_path='./figs/force_plot.jpg'):
    
    idx = data[split]['id2idx'][ID]
    new_idx = data[split]['idx2newidx'][idx]
    plt.rcParams.update({'font.size': 20})

    x_data = data[split]['X'].loc[idx,:]
    
    fig = shap.force_plot(explainer.expected_value,
                    shap_vals[new_idx,:],
                    x_data.apply(round_float), 
                    show=False, 
                    feature_names=[fill(code2name[x],15) for x in data[split]['X'].columns],
                    matplotlib=True, 
                    text_rotation=.01)
    for child in fig.axes[0].get_children():
        if isinstance(child, matplotlib.text.Text):
            child.set_fontsize(13)
            child.set_fontname('Arial')
            child.set_y(child._y*1.2)

    plt.savefig(save_path, bbox_inches='tight')
    return

def plot_and_save_summary_plot(shap_values, data_X, code2name, save_path):
    fig = plt.figure(figsize=(20,20))
    ax = plt.gca()
    plt.rcParams.update({'font.size': 22})
    shap.summary_plot(shap_values, data_X, show=False, max_display=10, plot_type="dot", alpha=0.25, color_bar=False)
    m = cm.ScalarMappable(cmap=red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=100, ax=ax)
    cb.outline.set_visible(False)
    cb.set_label(label='Feature Value', size=20)
    cb.set_ticklabels(['Low','High'])
    cb.ax.tick_params(labelsize=18, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    font = {'family' : 'Arial'}
    plt.rc('font', **font)
    ylabels=code2name
    ax.set_yticklabels([ylabels[x.get_text()] for x in ax.get_yticklabels()])
    ax.set_xlabel('SHAP value (impact on prediction)', fontsize = 20.0)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(save_path, bbox_inches='tight')
    return

def get_explainers_calibrated_clf(calibrated_clf, data):
    explainers = []
    for clf in calibrated_clf.calibrated_classifiers_:
        explainer = shap.TreeExplainer(clf.base_estimator, data=data, model_output="probability")
        explainers.append(explainer)
    return explainers

def get_shap_values_from_explainers(explainers, data):
    shap_vals = []
    for e in explainers:
        sv = e.shap_values(data)
        shap_vals.append(sv)
    return np.array(shap_vals).sum(axis=0) / len(shap_vals)
def main():
    #load data and models
    data_path_1y = DATA_DIR+'IHD_8139_1y_fts.csv'
    fts_1y = MODEL_DIR+'final_fts_1y_model.csv'
    data_1y = load_data(data_path_1y, cci_path_1y, fts_1y)
    
    data_path_5y = DATA_DIR+'IHD_8139_5y_fts.csv'
    fts_5y = MODEL_DIR+'final_fts_5y_model.csv'
    data_5y = load_data(data_path_5y, cci_path_5y, fts_5y)

    
    best_clin_1y = joblib.load(MODEL_DIR+"/best_ontology_fts_xgboost_1y.pkl")
    best_clin_5y = joblib.load(MODEL_DIR+"/best_ontology_fts_xgboost_5y.pkl")
        
    variable_list = pd.read_csv(DATA_DIR + "variable_list.csv")
    variable_list['readable_name+'] = variable_list.apply(lambda x: x['readable_name'] + ' (' + x['category'] + ')', axis=1)
    code2name = variable_list[['code','readable_name+']].set_index('code').T.to_dict('readable_name+')[0]
    code2name['CCI'] = 'Charlson Comorbidity Index'
    
    explainer_1y = shap.TreeExplainer(model=best_clin_1y, data=data_1y['train']['X'].values.astype(np.float), model_output="probability")
    shap_values_1y = explainer_1y.shap_values(data_1y['train']['X'])
    plot_and_save_summary_plot(shap_values_1y, data_1y['train']['X'], code2name, save_path='./figs/1y_top10_SHAPS.jpg')
       
    explainer_5y = shap.TreeExplainer(model=best_clin_5y, data=data_5y['train']['X'].values.astype(np.float), model_output="probability")
    shap_values_5y = explainer_5y.shap_values(data_5y['train']['X'])
    plot_and_save_summary_plot(shap_values_5y, data_5y['train']['X'], code2name, save_path='./figs/5y_top10_SHAPS.jpg')
    
    shap_values_5y_test = explainer_5y.shap_values(data_5y['test']['X'])

    low_pce_low_fusion_control = 'Q_gup2trpcO5HMY5pXYIf+tJTRplspO7xZDzVNKWPWs='
    high_pce_high_fusion_case = 'HmoHvTA_ETDF8C4nZosPHxdTlpOFouYTuwilP3IKITk='
    high_pce_low_fusion_control = 'MVSmFlBw8_JKNNZJPC7ngN7dOXe3Bv6jUxl30k+CkrQ='
    low_pce_high_fusion_case = '+QUuphnLmW3oFMe2VLPOmMUIaLqeACTxAIyt8R_LCx4='


    make_force_plot(low_pce_low_fusion_control, 'test',data_5y, explainer_5y, shap_values_5y_test, code2name, save_path='./figs/Fig4b_0_control_low_pce_low_fusion.jpg')
    make_force_plot(high_pce_high_fusion_case, 'test',data_5y, explainer_5y, shap_values_5y_test, code2name, save_path='./figs/Fig4b_1_case_high_pce_high_fusion.jpg')
    make_force_plot(high_pce_low_fusion_control, 'test',data_5y, explainer_5y, shap_values_5y_test, code2name, save_path='./figs/Fig4b_2_control_high_pce_low_fusion.jpg')
    make_force_plot(low_pce_high_fusion_case, 'test',data_5y, explainer_5y, shap_values_5y_test, code2name, save_path='./figs/Fig4b_3_case_low_pce_high_fusion.jpg')
if __name__=='__main__':
    main()
