from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def getAUROC(y_vals, y_scores):
    """
    Returns ROC curve values and AUROC for a set of labels (y_vals) and scores (y_scores)
    in numpy format
    """
    curve = {}
    AUROC = {}
    curve['fpr'], curve['tpr'], _ = roc_curve(y_vals, y_scores)
    AUROC['AUROC'] = roc_auc_score(y_vals, y_scores)
    return curve, AUROC
