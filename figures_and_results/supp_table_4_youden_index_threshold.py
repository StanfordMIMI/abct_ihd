from ml_models_utils import *
from fig2_aucs_pvals import get_predictions


def determineThreshold(labels, probs, method='Youden', cutoff=None):
    fpr, tpr, thresholds = roc_curve(labels,probs)
    if method == 'Youden':
        optimal_idx = np.argmax(tpr-fpr)
    elif method == 'Sensitivity':
        optimal_idx = np.argmin(np.abs(tpr - cutoff))
        preds = binarizeLabel(probs, thresholds[optimal_idx])
    elif method == 'Specificity':
        optimal_idx = np.argmin(np.abs((1-fpr)-cutoff))
    else:
        print('Incorrect method specified')
        return

    return thresholds[optimal_idx]

def binarizeLabel(predictions, cutoff=0.5):
    return [1 if x>=cutoff else 0 for x in predictions]
    
def getSensitivityAndSpecificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    return sensitivity, specificity

def getPPVNPV(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tp/(tp+fp),  tn / (fn+tn)

def main():

    preds_dir = '/PATH_TO/predictions/'
    predictions_1y_test = get_predictions(preds_dir+'IHD_8139_preds_all_1y.csv','test')
    predictions_5y_test = get_predictions(preds_dir+'IHD_8139_preds_all_5y.csv','test')

    predictions_1y_val = get_predictions(preds_dir+'IHD_8139_preds_all_1y.csv','val')
    predictions_5y_val = get_predictions(preds_dir+'IHD_8139_preds_all_5y.csv','val')

    pce_int_risk_1y = binarizeLabel(predictions_1y_test['pce_risk'], 0.075)
    pce_high_risk_1y = binarizeLabel(predictions_1y_test['pce_risk'], 0.2)
    pce_int_risk_5y = binarizeLabel(predictions_5y_test['pce_risk'], 0.075)
    pce_high_risk_5y = binarizeLabel(predictions_5y_test['pce_risk'], .2)

    
    pred_cols = ['seg_risk', 'clin_pred', 'img_pred', 'img_clin_fusion_preds']
    model_names = {'seg_risk':'Segmentation only', 'clin_pred':'  Clinical only  ', 'img_pred':'  Imaging only  ', 'img_clin_fusion_preds':'Imaging+Clinical'}
    threshold_methods = ['Youden']
    # BASELINE MODELS
    metrics_1y = {m:{tm:{} for tm in threshold_methods} for m in pred_cols+['pce_risk']}
    metrics_5y = {m:{tm:{} for tm in threshold_methods} for m in pred_cols+['pce_risk']}

    metrics_1y['pce_risk']['sensitivity_intermediate'], metrics_1y['pce_risk']['specificity_intermediate'] = getSensitivityAndSpecificity(predictions_1y_test['label'], pce_int_risk_1y)
    metrics_1y['pce_risk']['sensitivity_high'], metrics_1y['pce_risk']['specificity_high'] = getSensitivityAndSpecificity(predictions_1y_test['label'], pce_high_risk_1y)
    metrics_5y['pce_risk']['sensitivity_intermediate'], metrics_5y['pce_risk']['specificity_intermediate'] = getSensitivityAndSpecificity(predictions_5y_test['label'], pce_int_risk_5y)
    metrics_5y['pce_risk']['sensitivity_high'], metrics_5y['pce_risk']['specificity_high'] = getSensitivityAndSpecificity(predictions_5y_test['label'], pce_high_risk_5y)

    metrics_1y['pce_risk']['PPV_intermediate'], metrics_1y['pce_risk']['NPV_intermediate'] = getPPVNPV(predictions_1y_test['label'], pce_int_risk_1y)
    metrics_1y['pce_risk']['PPV_high'], metrics_1y['pce_risk']['NPV_high'] = getPPVNPV(predictions_1y_test['label'], pce_high_risk_1y)
    metrics_5y['pce_risk']['PPV_intermediate'], metrics_5y['pce_risk']['NPV_intermediate'] = getPPVNPV(predictions_5y_test['label'], pce_int_risk_5y)
    metrics_5y['pce_risk']['PPV_high'], metrics_5y['pce_risk']['NPV_high'] = getPPVNPV(predictions_5y_test['label'], pce_high_risk_5y)

    # PROPOSED MODELS
    thresholds_1y = {method:{} for method in threshold_methods}
    thresholds_5y = {method:{} for method in threshold_methods}
    for col in pred_cols:
        thresholds_1y['Youden'][col] = determineThreshold(predictions_1y_val['label'], predictions_1y_val[col])
        for thresh in threshold_methods:
            binary_preds = binarizeLabel(predictions_1y_test[col], cutoff=thresholds_1y[thresh][col])
            metrics_1y[col][thresh]['sensitivity'], metrics_1y[col][thresh]['specificity'] = getSensitivityAndSpecificity(predictions_1y_test['label'], binary_preds)
            metrics_1y[col][thresh]['PPV'], metrics_1y[col][thresh]['NPV'] = getPPVNPV(predictions_1y_test['label'], binary_preds)

        thresholds_5y['Youden'][col] = determineThreshold(predictions_5y_test['label'], predictions_5y_test[col])
        for thresh in threshold_methods:
            binary_preds = binarizeLabel(predictions_5y_test[col], cutoff=thresholds_5y[thresh][col])
            metrics_5y[col][thresh]['sensitivity'], metrics_5y[col][thresh]['specificity'] = getSensitivityAndSpecificity(predictions_5y_test['label'], binary_preds)
            metrics_5y[col][thresh]['PPV'], metrics_5y[col][thresh]['NPV'] = getPPVNPV(predictions_5y_test['label'], binary_preds)
    # print(metrics_1y)
    # print(metrics_5y)
    print(f"Model\t\t\tSe 1y\tSp 1y\tPPV 1y\tNPV 1y\t\tSe 5y\tSp 5y\tPPV 5y\tNPV 5y")
    print(f"PCE >7.5%\t\t{metrics_1y['pce_risk']['sensitivity_intermediate']*100:.1f}\t{metrics_1y['pce_risk']['specificity_intermediate']*100:.1f}\t{metrics_1y['pce_risk']['PPV_intermediate']*100:.1f}\t{metrics_1y['pce_risk']['NPV_intermediate']*100:.1f}\t\t{metrics_5y['pce_risk']['sensitivity_intermediate']*100:.1f}\t{metrics_5y['pce_risk']['specificity_intermediate']*100:.1f}\t{metrics_1y['pce_risk']['PPV_intermediate']*100:.1f}\t{metrics_5y['pce_risk']['NPV_intermediate']*100:.1f}")
    print(f"PCE >20%\t\t{metrics_1y['pce_risk']['sensitivity_high']*100:.1f}\t{metrics_1y['pce_risk']['specificity_high']*100:.1f}\t{metrics_1y['pce_risk']['PPV_high']*100:.1f}\t{metrics_1y['pce_risk']['NPV_high']*100:.1f}\t\t{metrics_5y['pce_risk']['sensitivity_high']*100:.1f}\t{metrics_5y['pce_risk']['specificity_high']*100:.1f}\t{metrics_1y['pce_risk']['PPV_high']*100:.1f}\t{metrics_5y['pce_risk']['NPV_high']*100:.1f}")
    print(f"S Only\t\t\t{metrics_1y['seg_risk']['Youden']['sensitivity']*100:.1f}\t{metrics_1y['seg_risk']['Youden']['specificity']*100:.1f}\t{metrics_1y['seg_risk']['Youden']['PPV']*100:.1f}\t{metrics_1y['seg_risk']['Youden']['NPV']*100:.1f}\t\t{metrics_5y['seg_risk']['Youden']['sensitivity']*100:.1f}\t{metrics_5y['seg_risk']['Youden']['specificity']*100:.1f}\t{metrics_5y['seg_risk']['Youden']['PPV']*100:.1f}\t{metrics_5y['seg_risk']['Youden']['NPV']*100:.1f}")
    print(f"C Only\t\t\t{metrics_1y['clin_pred']['Youden']['sensitivity']*100:.1f}\t{metrics_1y['clin_pred']['Youden']['specificity']*100:.1f}\t{metrics_1y['clin_pred']['Youden']['PPV']*100:.1f}\t{metrics_1y['clin_pred']['Youden']['NPV']*100:.1f}\t\t{metrics_5y['clin_pred']['Youden']['sensitivity']*100:.1f}\t{metrics_5y['clin_pred']['Youden']['specificity']*100:.1f}\t{metrics_5y['clin_pred']['Youden']['PPV']*100:.1f}\t{metrics_5y['clin_pred']['Youden']['NPV']*100:.1f}")
    print(f"I Only\t\t\t{metrics_1y['img_pred']['Youden']['sensitivity']*100:.1f}\t{metrics_1y['img_pred']['Youden']['specificity']*100:.1f}\t{metrics_1y['img_pred']['Youden']['PPV']*100:.1f}\t{metrics_1y['img_pred']['Youden']['NPV']*100:.1f}\t\t{metrics_5y['img_pred']['Youden']['sensitivity']*100:.1f}\t{metrics_5y['img_pred']['Youden']['specificity']*100:.1f}\t{metrics_5y['img_pred']['Youden']['PPV']*100:.1f}\t{metrics_5y['img_pred']['Youden']['NPV']*100:.1f}")
    print(f"I + C Fusion\t\t{metrics_1y['img_clin_fusion_preds']['Youden']['sensitivity']*100:.1f}\t{metrics_1y['img_clin_fusion_preds']['Youden']['specificity']*100:.1f}\t{metrics_1y['img_clin_fusion_preds']['Youden']['PPV']*100:.1f}\t{metrics_1y['img_clin_fusion_preds']['Youden']['NPV']*100:.1f}\t\t{metrics_5y['img_clin_fusion_preds']['Youden']['sensitivity']*100:.1f}\t{metrics_5y['img_clin_fusion_preds']['Youden']['specificity']*100:.1f}\t{metrics_5y['img_clin_fusion_preds']['Youden']['PPV']*100:.1f}\t{metrics_5y['img_clin_fusion_preds']['Youden']['NPV']*100:.1f}")
        

if __name__=='__main__':
    main()