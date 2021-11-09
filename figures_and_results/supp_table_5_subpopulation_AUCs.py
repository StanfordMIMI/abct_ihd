from fig2_aucs_pvals import get_predictions
from ml_models_utils import *
import logging

N_BOOTSTRAPS = 2000


def getAUROCswithCI(predictions, preds_columns):
    def getAUROCwithCI(labels, preds, nBootstraps):
        auc = roc_auc_score(labels, preds)
        lower_ci, upper_ci = bootStrapModelAUROC(labels, preds, numBootstraps=nBootstraps)
        return {'auroc':auc, 'ci_low':lower_ci, 'ci_high':upper_ci}
    # PCE
    auroc = {}
    for i, clf_1 in enumerate(preds_columns):
        auroc[clf_1] = getAUROCwithCI(predictions['label'].values, predictions[clf_1].values, N_BOOTSTRAPS) 
    
    return auroc

def getAUPRCswithCI(predictions, preds_columns):
    def getAUPRCwithCI(labels, preds, nBootstraps):
        auc = average_precision_score(labels, preds)
        lower_ci, upper_ci = bootStrapModelAUPRC(labels, preds, numBootstraps=nBootstraps)
        return {'auprc':auc, 'ci_low':lower_ci, 'ci_high':upper_ci}
    # PCE
    auprc = {}
    for i, clf_1 in enumerate(preds_columns):
        auprc[clf_1] = getAUPRCwithCI(predictions['label'].values, predictions[clf_1].values, N_BOOTSTRAPS) 
    return auprc

def log_AUCs(auroc1y, auprc1y, auroc5y, aurprc5y, preds_columns, preds_names):
    logging.info(f"1y AUROC (95% CI)\t\t\t\t1y AUPRC (95% CI)\t\t\t\t5y AUROC (95% CI)\t\t\t\t5y AUPRC (95% CI)\t\t\t\tModel")
    for i,preds_name in enumerate(preds_names):
        logging.info(f"""{auroc1y[preds_columns[i]]['auroc']:.2f} ({auroc1y[preds_columns[i]]['ci_low']:.2f}-{auroc1y[preds_columns[i]]['ci_high']:.2f})\t\
              {auprc1y[preds_columns[i]]['auprc']:.2f} ({auprc1y[preds_columns[i]]['ci_low']:.2f}-{auprc1y[preds_columns[i]]['ci_high']:.2f})\t \
              {auroc5y[preds_columns[i]]['auroc']:.2f} ({auroc5y[preds_columns[i]]['ci_low']:.2f}-{auroc5y[preds_columns[i]]['ci_high']:.2f})\t \
              {aurprc5y[preds_columns[i]]['auprc']:.2f} ({aurprc5y[preds_columns[i]]['ci_low']:.2f}-{aurprc5y[preds_columns[i]]['ci_high']:.2f})\t\
              {preds_name}""")
    return

def get_overall_aucs(predictions_1y, predictions_5y, preds_columns, preds_names):
    auroc_1y = getAUROCswithCI(predictions_1y, preds_columns)
    auroc_5y = getAUROCswithCI(predictions_5y, preds_columns)
    auprc_1y = getAUPRCswithCI(predictions_1y, preds_columns)
    auprc_5y = getAUPRCswithCI(predictions_5y, preds_columns)

    #logging
    logging.info("Overall model performance")
    log_AUCs(auroc_1y, auprc_1y, auroc_5y, auprc_5y, preds_columns, preds_names)
    return auroc_1y, auroc_5y, auprc_1y, auprc_5y

def get_percent_positive(df):
    return f"{100*df[df['label']==1].shape[0]/df.shape[0]:.1f}"

def get_test_cohort_data(data_dir = '/PATH_TO/data/'):
    test_cohort_data_1y = get_predictions(data_dir+'IHD_8139_1y_ft_matrix_full.csv', 'test')
    test_cohort_data_5y = get_predictions(data_dir+'IHD_8139_5y_ft_matrix_full.csv', 'test')
    return test_cohort_data_1y, test_cohort_data_5y 

def get_missing_nonmissing_model_performance(predictions_1y, predictions_5y, preds_columns, preds_names):
    test_cohort_data_1y, test_cohort_data_5y = get_test_cohort_data()

    test_1y_full_pce_data_ids = test_cohort_data_1y[(~pd.isna(test_cohort_data_1y['smoker']))&(~pd.isna(test_cohort_data_1y['latest_value_chol_total']))&(~pd.isna(test_cohort_data_1y['latest_value_chol_hdl'])) & (~pd.isna(test_cohort_data_1y['latest_value_systolic']))][['anon_id']]
    test_5y_full_pce_data_ids = test_cohort_data_5y[(~pd.isna(test_cohort_data_5y['smoker']))&(~pd.isna(test_cohort_data_5y['latest_value_chol_total']))&(~pd.isna(test_cohort_data_5y['latest_value_chol_hdl'])) & (~pd.isna(test_cohort_data_5y['latest_value_systolic']))][['anon_id']]

    test_1y_missing_pce_data_ids = pd.DataFrame({'anon_id':[x for x in [x for x in test_cohort_data_1y['anon_id']] if x not in set([x for x in test_1y_full_pce_data_ids['anon_id']])]})
    test_5y_missing_pce_data_ids = pd.DataFrame({'anon_id':[x for x in [x for x in test_cohort_data_5y['anon_id']] if x not in set([x for x in test_5y_full_pce_data_ids['anon_id']])]})
    
    preds_full_pce_data_1y = pd.merge(left=predictions_1y, right=test_1y_full_pce_data_ids, how='right')
    preds_full_pce_data_5y = pd.merge(left=predictions_5y, right=test_5y_full_pce_data_ids, how='right')
    preds_missing_pce_data_1y = pd.merge(left=predictions_1y, right=test_1y_missing_pce_data_ids, how='right')
    preds_missing_pce_data_5y = pd.merge(left=predictions_5y, right=test_5y_missing_pce_data_ids, how='right')
    
    auroc_1y_pce_complete = getAUROCswithCI(preds_full_pce_data_1y, preds_columns)
    auprc_1y_pce_complete = getAUPRCswithCI(preds_full_pce_data_1y, preds_columns)
    auroc_5y_pce_complete = getAUROCswithCI(preds_full_pce_data_5y, preds_columns)
    auprc_5y_pce_complete = getAUPRCswithCI(preds_full_pce_data_5y, preds_columns)

    auroc_1y_pce_missing = getAUROCswithCI(preds_missing_pce_data_1y, preds_columns)
    auprc_1y_pce_missing = getAUPRCswithCI(preds_missing_pce_data_1y, preds_columns)
    auroc_5y_pce_missing = getAUROCswithCI(preds_missing_pce_data_5y, preds_columns)
    auprc_5y_pce_missing = getAUPRCswithCI(preds_missing_pce_data_5y, preds_columns)

    #logging
    logging.info(f"\nComplete PCE data (1y n={preds_full_pce_data_1y.shape[0]} / 5y n={preds_full_pce_data_5y.shape[0]})")
    logging.info(f"Complete PCE data % pos(1y n={get_percent_positive(preds_full_pce_data_1y)} / 5y n={get_percent_positive(preds_full_pce_data_5y)})")
    log_AUCs(auroc_1y_pce_complete, auprc_1y_pce_complete, auroc_5y_pce_complete, auprc_5y_pce_complete, preds_columns, preds_names) 
    logging.info(f"\nmissing PCE data (1y n={preds_missing_pce_data_1y.shape[0]} / 5y n={preds_missing_pce_data_5y.shape[0]})")
    logging.info(f"missing PCE data % pos(1y n={get_percent_positive(preds_missing_pce_data_1y)} / 5y n={get_percent_positive(preds_missing_pce_data_5y)})")
    log_AUCs(auroc_1y_pce_missing, auprc_1y_pce_missing, auroc_5y_pce_missing, auprc_5y_pce_missing, preds_columns, preds_names)

    return auroc_1y_pce_complete, auprc_1y_pce_complete, auroc_5y_pce_complete, auprc_5y_pce_complete, \
            auroc_1y_pce_missing, auprc_1y_pce_missing, auroc_5y_pce_missing, auprc_5y_pce_missing

def get_age_split_performance(predictions_1y, predictions_5y, preds_columns, preds_names):
    test_cohort_data_1y, test_cohort_data_5y = get_test_cohort_data()
    test_1y_40_75_ids = test_cohort_data_1y[(test_cohort_data_1y['age_at_scan']>=40) & (test_cohort_data_1y['age_at_scan']<76)][['anon_id']]
    test_5y_40_75_ids = test_cohort_data_5y[(test_cohort_data_5y['age_at_scan']>=40) & (test_cohort_data_5y['age_at_scan']<76)][['anon_id']]

    test_1y_u40_ids = test_cohort_data_1y[(test_cohort_data_1y['age_at_scan']<40)][['anon_id']]
    test_5y_u40_ids = test_cohort_data_5y[(test_cohort_data_5y['age_at_scan']<40)][['anon_id']]

    test_1y_o_75_ids = pd.DataFrame({'anon_id':[x for x in [x for x in test_cohort_data_1y['anon_id']] if x not in set([x for x in test_1y_40_75_ids['anon_id']] + [x for x in test_1y_u40_ids['anon_id']])]})
    test_5y_o_75_ids = pd.DataFrame({'anon_id':[x for x in [x for x in test_cohort_data_5y['anon_id']] if x not in set([x for x in test_5y_40_75_ids['anon_id']] + [x for x in test_5y_u40_ids['anon_id']])]})

    preds_40_75_1y = pd.merge(left=predictions_1y, right=test_1y_40_75_ids, how='right')
    preds_40_75_5y = pd.merge(left=predictions_5y, right=test_5y_40_75_ids, how='right')
    preds_u40_1y = pd.merge(left=predictions_1y, right=test_1y_u40_ids, how='right')
    preds_u40_5y = pd.merge(left=predictions_5y, right=test_5y_u40_ids, how='right')
    preds_o75_1y = pd.merge(left=predictions_1y, right=test_1y_o_75_ids, how='right')
    preds_o75_5y = pd.merge(left=predictions_5y, right=test_5y_o_75_ids, how='right')

    auroc_1y_40_75 = getAUROCswithCI(preds_40_75_1y, preds_columns)
    auprc_1y_40_75 = getAUPRCswithCI(preds_40_75_1y, preds_columns)
    auroc_5y_40_75 = getAUROCswithCI(preds_40_75_5y, preds_columns)
    auprc_5y_40_75 = getAUPRCswithCI(preds_40_75_5y, preds_columns)

    auroc_1y_u40 = getAUROCswithCI(preds_u40_1y, preds_columns)
    auprc_1y_u40 = getAUPRCswithCI(preds_u40_1y, preds_columns)
    auroc_5y_u40 = getAUROCswithCI(preds_u40_5y, preds_columns)
    auprc_5y_u40 = getAUPRCswithCI(preds_u40_5y, preds_columns)

    auroc_1y_o75 = getAUROCswithCI(preds_o75_1y, preds_columns)
    auprc_1y_o75 = getAUPRCswithCI(preds_o75_1y, preds_columns)
    auroc_5y_o75 = getAUROCswithCI(preds_o75_5y, preds_columns)
    auprc_5y_o75 = getAUPRCswithCI(preds_o75_5y, preds_columns)

    #logging
    logging.info(f"\n 40-75 y/o (1y n={preds_40_75_1y.shape[0]} / 5y n={preds_40_75_5y.shape[0]})")
    logging.info(f"40-75 y/o % pos(1y n={get_percent_positive(preds_40_75_1y)} / 5y n={get_percent_positive(preds_40_75_5y)})")
    log_AUCs(auroc_1y_40_75, auprc_1y_40_75, auroc_5y_40_75, auprc_5y_40_75, preds_columns, preds_names)
    logging.info(f"\n Under 40 y/o (n={preds_u40_1y.shape[0]} / 5y n={preds_u40_5y.shape[0]})")
    logging.info(f"Under 40 y/o % pos(1y n={get_percent_positive(preds_u40_1y)} / 5y n={get_percent_positive(preds_u40_5y)})")
    log_AUCs(auroc_1y_u40, auprc_1y_u40, auroc_5y_u40, auprc_5y_u40, preds_columns, preds_names)
    logging.info(f"\n Over 75 y/o (n={preds_o75_1y.shape[0]} / 5y n={preds_o75_5y.shape[0]})")
    logging.info(f"Over 75 y/o % pos(1y n={get_percent_positive(preds_o75_1y)} / 5y n={get_percent_positive(preds_o75_5y)})")
    log_AUCs(auroc_1y_o75, auprc_1y_o75, auroc_5y_o75, auprc_5y_o75, preds_columns, preds_names)

    return auroc_1y_40_75, auprc_1y_40_75, auroc_5y_40_75, auprc_5y_40_75, \
            auroc_1y_u40, auprc_1y_u40, auroc_5y_u40, auprc_5y_u40, \
            auroc_1y_o75, auprc_1y_o75, auroc_5y_o75, auprc_5y_o75

def get_gender_performance(predictions_1y, predictions_5y, preds_columns, preds_names):
    test_cohort_data_1y, test_cohort_data_5y = get_test_cohort_data()
    test_1y_male_ids = test_cohort_data_1y[test_cohort_data_1y['gender'] == True][['anon_id']]
    test_1y_female_ids = test_cohort_data_1y[test_cohort_data_1y['gender'] == False][['anon_id']]

    test_5y_male_ids = test_cohort_data_5y[test_cohort_data_5y['gender'] == True][['anon_id']]
    test_5y_female_ids = test_cohort_data_5y[test_cohort_data_5y['gender'] == False][['anon_id']]

    preds_male_1y = pd.merge(left=predictions_1y, right=test_1y_male_ids, how='right')
    preds_male_5y = pd.merge(left=predictions_5y, right=test_5y_male_ids, how='right')
    preds_female_1y = pd.merge(left=predictions_1y, right=test_1y_female_ids, how='right')
    preds_female_5y = pd.merge(left=predictions_5y, right=test_5y_female_ids, how='right')

    auroc_1y_male = getAUROCswithCI(preds_male_1y, preds_columns)
    auprc_1y_male = getAUPRCswithCI(preds_male_1y, preds_columns)
    auroc_5y_male = getAUROCswithCI(preds_male_5y, preds_columns)
    auprc_5y_male = getAUPRCswithCI(preds_male_5y, preds_columns)

    auroc_1y_female = getAUROCswithCI(preds_female_1y, preds_columns)
    auprc_1y_female = getAUPRCswithCI(preds_female_1y, preds_columns)
    auroc_5y_female = getAUROCswithCI(preds_female_5y, preds_columns)
    auprc_5y_female = getAUPRCswithCI(preds_female_5y, preds_columns)

    logging.info(f"\n Male (1y n={preds_male_1y.shape[0]} / 5y n={preds_male_5y.shape[0]})")
    logging.info(f"Male % pos(1y n={get_percent_positive(preds_male_1y)} / 5y n={get_percent_positive(preds_male_5y)})")
    log_AUCs(auroc_1y_male, auprc_1y_male, auroc_5y_male, auprc_5y_male, preds_columns, preds_names)

    logging.info(f"\n Female (1y n={preds_female_1y.shape[0]} / 5y n={preds_female_5y.shape[0]})")
    logging.info(f"Female  % pos(1y n={get_percent_positive(preds_female_1y)} / 5y n={get_percent_positive(preds_female_5y)})")
    log_AUCs(auroc_1y_female, auprc_1y_female, auroc_5y_female, auprc_5y_female, preds_columns, preds_names)

    return auroc_1y_male, auprc_1y_male, auroc_5y_male, auprc_5y_male, \
            auroc_1y_female, auprc_1y_female, auroc_5y_female, auprc_5y_female

def get_race_eth_performance(predictions_1y, predictions_5y, preds_columns, preds_names, data_dir='/PATH_TO/data/'):
    race_data = pd.read_csv(data_dir+'IHD_8139_race_eth.csv')
    race_data_test_1y = race_data.merge(predictions_1y, left_on='anon_id', right_on='anon_id', how='right')
    race_data_test_5y = race_data.merge(predictions_5y, left_on='anon_id', right_on='anon_id', how='right')
    
    preds_1y_asian = race_data_test_1y[race_data_test_1y['race_eth']=='Asian']
    preds_1y_black = race_data_test_1y[race_data_test_1y['race_eth']=='Black']
    preds_1y_hispanic = race_data_test_1y[race_data_test_1y['race_eth']=='Hispanic']
    preds_1y_other = race_data_test_1y[race_data_test_1y['race_eth']=='Other']
    preds_1y_white = race_data_test_1y[race_data_test_1y['race_eth']=='White']

    preds_5y_asian = race_data_test_5y[race_data_test_5y['race_eth']=='Asian']
    preds_5y_black = race_data_test_5y[race_data_test_5y['race_eth']=='Black']
    preds_5y_hispanic = race_data_test_5y[race_data_test_5y['race_eth']=='Hispanic']
    preds_5y_other = race_data_test_5y[race_data_test_5y['race_eth']=='Other']
    preds_5y_white = race_data_test_5y[race_data_test_5y['race_eth']=='White']

    auroc_1y_asian, auprc_1y_asian = getAUROCswithCI(preds_1y_asian, preds_columns), getAUPRCswithCI(preds_1y_asian, preds_columns)
    auroc_1y_black, auprc_1y_black = getAUROCswithCI(preds_1y_black, preds_columns), getAUPRCswithCI(preds_1y_black, preds_columns)
    auroc_1y_hispanic, auprc_1y_hispanic = getAUROCswithCI(preds_1y_hispanic, preds_columns), getAUPRCswithCI(preds_1y_hispanic, preds_columns)
    auroc_1y_other, auprc_1y_other = getAUROCswithCI(preds_1y_other, preds_columns), getAUPRCswithCI(preds_1y_other, preds_columns)
    auroc_1y_white, auprc_1y_white = getAUROCswithCI(preds_1y_white, preds_columns), getAUPRCswithCI(preds_1y_white, preds_columns)
    
    auroc_5y_asian, auprc_5y_asian = getAUROCswithCI(preds_5y_asian, preds_columns), getAUPRCswithCI(preds_5y_asian, preds_columns)
    auroc_5y_black, auprc_5y_black = getAUROCswithCI(preds_5y_black, preds_columns), getAUPRCswithCI(preds_5y_black, preds_columns)
    auroc_5y_hispanic, auprc_5y_hispanic = getAUROCswithCI(preds_5y_hispanic, preds_columns), getAUPRCswithCI(preds_5y_hispanic, preds_columns)
    auroc_5y_other, auprc_5y_other = getAUROCswithCI(preds_5y_other, preds_columns), getAUPRCswithCI(preds_5y_other, preds_columns)
    auroc_5y_white, auprc_5y_white = getAUROCswithCI(preds_5y_white, preds_columns), getAUPRCswithCI(preds_5y_white, preds_columns)

    # logging

    logging.info(f"\n Asian (1y n={preds_1y_asian.shape[0]} / 5y n={preds_5y_asian.shape[0]})")
    logging.info(f"Asian  % pos(1y n={get_percent_positive(preds_1y_asian)} / 5y n={get_percent_positive(preds_5y_asian)})")
    log_AUCs(auroc_1y_asian, auprc_1y_asian, auroc_5y_asian, auprc_5y_asian, preds_columns, preds_names)

    logging.info(f"\n Black (1y n={preds_1y_black.shape[0]} / 5y n={preds_5y_black.shape[0]})")
    logging.info(f"Black  % pos(1y n={get_percent_positive(preds_1y_black)} / 5y n={get_percent_positive(preds_5y_black)})")
    log_AUCs(auroc_1y_black, auprc_1y_black, auroc_5y_black, auprc_5y_black, preds_columns, preds_names)

    logging.info(f"\n Hispanic (1y n={preds_1y_hispanic.shape[0]} / 5y n={preds_5y_hispanic.shape[0]})")
    logging.info(f"Hispanic  % pos(1y n={get_percent_positive(preds_1y_hispanic)} / 5y n={get_percent_positive(preds_5y_hispanic)})")
    log_AUCs(auroc_1y_hispanic, auprc_1y_hispanic, auroc_5y_hispanic, auprc_5y_hispanic, preds_columns, preds_names)

    logging.info(f"\n Other (1y n={preds_1y_other.shape[0]} / 5y n={preds_5y_other.shape[0]})")
    logging.info(f"Other  % pos(1y n={get_percent_positive(preds_1y_other)} / 5y n={get_percent_positive(preds_5y_other)})")
    log_AUCs(auroc_1y_other, auprc_1y_other, auroc_5y_other, auprc_5y_other, preds_columns, preds_names)

    logging.info(f"\n White (1y n={preds_1y_white.shape[0]} / 5y n={preds_5y_white.shape[0]})")
    logging.info(f"White  % pos(1y n={get_percent_positive(preds_1y_white)} / 5y n={get_percent_positive(preds_5y_white)})")
    log_AUCs(auroc_1y_white, auprc_1y_white, auroc_5y_white, auprc_5y_white, preds_columns, preds_names)

    return auroc_1y_asian, auprc_1y_asian, auroc_5y_asian, auprc_5y_asian,\
            auroc_1y_black, auprc_1y_black, auroc_5y_black, auprc_5y_black,\
            auroc_1y_hispanic, auprc_1y_hispanic, auroc_5y_hispanic, auprc_5y_hispanic,\
            auroc_1y_other, auprc_1y_other, auroc_5y_other, auprc_5y_other,\
            auroc_1y_white, auprc_1y_white, auroc_5y_white, auprc_5y_white
def get_lipid_lowering_performance(predictions_1y, predictions_5y, preds_columns, preds_names):
    test_cohort_data_1y, test_cohort_data_5y = get_test_cohort_data()
    test_1y_llm_ids = test_cohort_data_1y[test_cohort_data_1y['C01'] > 0][['anon_id']]
    test_1y_nllm_ids = test_cohort_data_1y[test_cohort_data_1y['C01'] == 0][['anon_id']]

    test_5y_llm_ids = test_cohort_data_5y[test_cohort_data_5y['C01'] > 0][['anon_id']]
    test_5y_nllm_ids = test_cohort_data_5y[test_cohort_data_5y['C01'] == 0][['anon_id']]

    preds_llm_1y = pd.merge(left=predictions_1y, right=test_1y_llm_ids, how='right')
    preds_llm_5y = pd.merge(left=predictions_5y, right=test_5y_llm_ids, how='right')
    preds_nllm_1y = pd.merge(left=predictions_1y, right=test_1y_nllm_ids, how='right')
    preds_nllm_5y = pd.merge(left=predictions_5y, right=test_5y_nllm_ids, how='right')

    auroc_1y_llm = getAUROCswithCI(preds_llm_1y, preds_columns)
    auprc_1y_llm = getAUPRCswithCI(preds_llm_1y, preds_columns)
    auroc_5y_llm = getAUROCswithCI(preds_llm_5y, preds_columns)
    auprc_5y_llm = getAUPRCswithCI(preds_llm_5y, preds_columns)

    auroc_1y_nllm = getAUROCswithCI(preds_nllm_1y, preds_columns)
    auprc_1y_nllm = getAUPRCswithCI(preds_nllm_1y, preds_columns)
    auroc_5y_nllm = getAUROCswithCI(preds_nllm_5y, preds_columns)
    auprc_5y_nllm = getAUPRCswithCI(preds_nllm_5y, preds_columns)

    logging.info(f"\n Taking Lipid Lowering (1y n={preds_llm_1y.shape[0]} / 5y n={preds_llm_5y.shape[0]})")
    logging.info(f"Taking Lipid Lowering % pos(1y n={get_percent_positive(preds_llm_1y)} / 5y n={get_percent_positive(preds_llm_5y)})")
    log_AUCs(auroc_1y_llm, auprc_1y_llm, auroc_5y_llm, auprc_5y_llm, preds_columns, preds_names)

    logging.info(f"\n not Taking Lipid Lowering (1y n={preds_nllm_1y.shape[0]} / 5y n={preds_nllm_5y.shape[0]})")
    logging.info(f"notTaking Lipid Lowering  % pos(1y n={get_percent_positive(preds_nllm_1y)} / 5y n={get_percent_positive(preds_nllm_5y)})")
    log_AUCs(auroc_1y_nllm, auprc_1y_nllm, auroc_5y_nllm, auprc_5y_nllm, preds_columns, preds_names)

    return auroc_1y_llm, auprc_1y_llm, auroc_5y_llm, auprc_5y_llm, \
            auroc_1y_nllm, auprc_1y_nllm, auroc_5y_nllm, auprc_5y_nllm

def get_acute_stable_performance(predictions_1y, predictions_5y, preds_columns, preds_names, acute_data_dir='/PATH_TO/data/'):
    acute_events_data = pd.read_csv(acute_data_dir+'IHD_8139_acute_events.csv')
    
    test_1y_acute_ids = acute_events_data[(acute_events_data['had_acute_event_1y'] ==True) | (pd.isna(acute_events_data['had_acute_event_1y']))][['anon_id']].merge(predictions_1y[['anon_id']])
    test_1y_nonacute_ids = acute_events_data[(acute_events_data['had_acute_event_1y'] ==False) | (pd.isna(acute_events_data['had_acute_event_1y']))][['anon_id']].merge(predictions_1y[['anon_id']])

    test_5y_acute_ids = acute_events_data[(acute_events_data['had_acute_event_5y'] ==True) | (pd.isna(acute_events_data['had_acute_event_5y']))][['anon_id']].merge(predictions_5y[['anon_id']])
    test_5y_nonacute_ids = acute_events_data[(acute_events_data['had_acute_event_5y'] ==False) | (pd.isna(acute_events_data['had_acute_event_5y']))][['anon_id']].merge(predictions_5y[['anon_id']])

    preds_acute_1y = pd.merge(left=predictions_1y, right=test_1y_acute_ids, how='inner')
    preds_acute_5y = pd.merge(left=predictions_5y, right=test_5y_acute_ids, how='inner')
    preds_nonacute_1y = pd.merge(left=predictions_1y, right=test_1y_nonacute_ids, how='inner')
    preds_nonacute_5y = pd.merge(left=predictions_5y, right=test_5y_nonacute_ids, how='inner')

    auroc_1y_acute = getAUROCswithCI(preds_acute_1y, preds_columns)
    auprc_1y_acute = getAUPRCswithCI(preds_acute_1y, preds_columns)
    auroc_5y_acute = getAUROCswithCI(preds_acute_5y, preds_columns)
    auprc_5y_acute = getAUPRCswithCI(preds_acute_5y, preds_columns)

    auroc_1y_nonacute = getAUROCswithCI(preds_nonacute_1y, preds_columns)
    auprc_1y_nonacute = getAUPRCswithCI(preds_nonacute_1y, preds_columns)
    auroc_5y_nonacute = getAUROCswithCI(preds_nonacute_5y, preds_columns)
    auprc_5y_nonacute = getAUPRCswithCI(preds_nonacute_5y, preds_columns)
    
    logging.info(f"\n Acute IHD (1y n={preds_acute_1y.shape[0]} / 5y n={preds_acute_5y.shape[0]})")
    logging.info(f"Acute IHD  % pos(1y n={get_percent_positive(preds_acute_1y)} / 5y n={get_percent_positive(preds_acute_5y)})")
    log_AUCs(auroc_1y_acute, auprc_1y_acute, auroc_5y_acute, auprc_5y_acute, preds_columns, preds_names)

    logging.info(f"\n Non-Acute IHD (1y n={preds_nonacute_1y.shape[0]} / 5y n={preds_nonacute_5y.shape[0]})")
    logging.info(f"Non-Acute IHD  % pos(1y n={get_percent_positive(preds_nonacute_1y)} / 5y n={get_percent_positive(preds_nonacute_5y)})")
    log_AUCs(auroc_1y_nonacute, auprc_1y_nonacute, auroc_5y_nonacute, auprc_5y_nonacute, preds_columns, preds_names)

    return auroc_1y_acute, auprc_1y_acute, auroc_5y_acute, auprc_5y_acute, \
            auroc_1y_nonacute, auprc_1y_nonacute, auroc_5y_nonacute, auprc_5y_nonacute
def main():
    logging.basicConfig(filename=str('../logs/')+str('supp_table_subpopulation_AUCs.log'), level=logging.INFO, format='%(message)s')
    
    preds_dir = '/PATH_TO/predictions/'
    predictions_1y = get_predictions(preds_dir+'IHD_8139_preds_all_1y.csv','test')
    predictions_5y = get_predictions(preds_dir+'IHD_8139_preds_all_5y.csv','test')


    preds_columns = ['frs', 'pce_risk', 'seg_risk', 'pce_seg_model_pred', 'clin_pred', 'img_pred', 'img_clin_fusion_preds', 'img_clin_seg_fusion_preds']
    preds_names = ['FRS', 'PCE', 'Segmentation', 'PCE+Segmentation', 'Clinical only', 'Imaging only', 'Imaging+Clinical Fusion', 'Imaging+Clinical+Segmentation Fusion']
    
    #overall
    auroc_1y, auroc_5y, auprc_1y, auprc_5y = get_overall_aucs(predictions_1y, predictions_5y, preds_columns, preds_names)
    
    #missing/not PCE covariates
    _ = get_missing_nonmissing_model_performance(predictions_1y, predictions_5y, preds_columns, preds_names)
    
    #40-75 y/o
    _ = get_age_split_performance(predictions_1y, predictions_5y, preds_columns, preds_names)

    #gender
    _ = get_gender_performance(predictions_1y, predictions_5y, preds_columns, preds_names)
    
    #race
    _ = get_race_eth_performance(predictions_1y, predictions_5y, preds_columns, preds_names)

    #lipid lowering drugs
    _ = get_lipid_lowering_performance(predictions_1y, predictions_5y, preds_columns, preds_names)
    
    #acute vs nonacute events
    _ = get_acute_stable_performance(predictions_1y, predictions_5y, preds_columns, preds_names)    
    
    
if __name__ == '__main__':
    main()