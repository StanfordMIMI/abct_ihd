from ml_models_utils import *
import logging

def getAUROCswithCI(predictions, preds_columns, preds_names):
    def getAUROCwithCI(labels, preds, nBootstraps):
        auc = roc_auc_score(labels, preds)
        lower_ci, upper_ci = bootStrapModelAUROC(labels, preds, numBootstraps=nBootstraps)
        return {'auroc':auc, 'ci_low':lower_ci, 'ci_high':upper_ci}
    
    auroc = {}
    for i, clf_1 in enumerate(preds_columns):
        auroc[clf_1] = getAUROCwithCI(predictions['label'].values, predictions[clf_1].values, 2000) 
    
    logging.info(f"AUROC (95% CI)\t\tModel")
    for i,preds_name in enumerate(preds_names):
        logging.info(f"{auroc[preds_columns[i]]['auroc']:.2f} ({auroc[preds_columns[i]]['ci_low']:.2f}-{auroc[preds_columns[i]]['ci_high']:.2f})\t{preds_name}")

    return auroc

def getAUPRCswithCI(predictions, preds_columns, preds_names):
    def getAUPRCwithCI(labels, preds, nBootstraps):
        auc = average_precision_score(labels, preds)
        lower_ci, upper_ci = bootStrapModelAUPRC(labels, preds, numBootstraps=nBootstraps)
        return {'auprc':auc, 'ci_low':lower_ci, 'ci_high':upper_ci}
    
    auprc = {}
    for i, clf_1 in enumerate(preds_columns):
        auprc[clf_1] = getAUPRCwithCI(predictions['label'].values, predictions[clf_1].values, 2000) 
    
    logging.info(f"AUPRC (95% CI)\t\tModel")
    for i,preds_name in enumerate(preds_names):
        logging.info(f"{auprc[preds_columns[i]]['auprc']:.2f} ({auprc[preds_columns[i]]['ci_low']:.2f}-{auprc[preds_columns[i]]['ci_high']:.2f})\t{preds_name}")
          
    return auprc

def get_bootstrap_p_vals(predictions, preds_columns):
    risks = {x:predictions.loc[:,x].values for x in preds_columns}
    labels = predictions['label'].values
    pvals= {x:{} for x in preds_columns}
    for i, risk1 in enumerate(preds_columns):
        for j, risk2 in enumerate(preds_columns):
            if j>i:
                pvals[risk1][risk2] = bootstrapPRCDiff(labels, risks[risk1], risks[risk2])
    
    logging.info(f"Models\t\t\t Bootstrap AUCPR diff P value")

    for k1 in pvals.keys():
        for k2 in pvals[k1].keys():
            logging.info(f"{k1} vs {k2}:\t {formatPValue(pvals[k1][k2])}")
    return pvals

def get_predictions(fpath, subset=None):
    preds = pd.read_csv(fpath, low_memory=False)
    if subset is not None:
        return preds[preds['set']==subset]
    return preds


def main():
    #set up logging; load data
    logging.basicConfig(filename=str('../logs/')+str('fig2_aucs_pvals.log'), level=logging.INFO)
    preds_dir = '/PATH_TO/predictions/'
    predictions_1y = get_predictions(preds_dir+'IHD_8139_preds_all_1y.csv','test')
    predictions_5y = get_predictions(preds_dir+'IHD_8139_preds_all_5y.csv','test')
    
    preds_columns = ['frs', 'pce_risk', 'seg_risk', 'pce_seg_model_pred', 'clin_pred', 'img_pred', 'img_clin_fusion_preds', 'img_clin_seg_fusion_preds']
    preds_names = ['FRS', 'PCE', 'Segmentation', 'PCE+Segmentation', 'Clinical only', 'Imaging only', 'Imaging+Clinical Fusion', 'Imaging+Clinical+Segmentation Fusion']

    logging.info('-'*50+'1y'+'-'*50)
    auroc_1y = getAUROCswithCI(predictions_1y, preds_columns, preds_names)
    auprc_1y = getAUPRCswithCI(predictions_1y, preds_columns, preds_names)
    oneypvals = get_bootstrap_p_vals(predictions_1y, preds_columns)
    
    logging.info('\n'+'-'*50+'5y'+'-'*50)
    
    auroc_5y = getAUROCswithCI(predictions_5y, preds_columns, preds_names)
    auprc_5y = getAUPRCswithCI(predictions_5y, preds_columns, preds_names)
    fiveypvals = get_bootstrap_p_vals(predictions_5y, preds_columns)
if __name__=='__main__':
    main()