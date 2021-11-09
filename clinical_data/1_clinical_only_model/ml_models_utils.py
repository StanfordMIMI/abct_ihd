from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning)# ignore all future warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def getMetrics(y_true, y_pred):
    "Returns sensitivity, specificity, precision, f1 score and accuracy for a set of true labels and predictions"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    f1 = 2*tp/ (2*tp+fp+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    return sensitivity, specificity, precision, f1, accuracy

def findOptimalThreshold(tpr, fpr, thresholds):
    "Returns optimal threshold for an ROC curve based on Youden's index"
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def bootStrapModelAUROC(y_true, y_pred, numBootstraps=1000):
    """
    Returns upper and lower 95% confidence intervals for a model by bootstrapping the true and predicted
    labels for a model.
    """
    bootstrapAUROCs = []
    rndom_state = np.random.RandomState(17)
    for i in range(numBootstraps):
        idx = rndom_state.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = roc_auc_score(y_true[idx],y_pred[idx])
        bootstrapAUROCs.append(score)
    sortedAUROCs = np.array(bootstrapAUROCs)
    sortedAUROCs.sort()
    lower_ci = sortedAUROCs[int(0.05*len(sortedAUROCs))]
    upper_ci = sortedAUROCs[int(0.95*len(sortedAUROCs))]
    return lower_ci, upper_ci

def bootStrapModelAUPRC(y_true, y_pred, numBootstraps=1000):
    """
    Returns upper and lower 95% confidence intervals for a model by bootstrapping the true and predicted
    labels for a model.
    """
    bootstrapAUPRCs = []
    rndom_state = np.random.RandomState(17)
    for i in range(numBootstraps):
        idx = rndom_state.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = average_precision_score(y_true[idx],y_pred[idx])
        bootstrapAUPRCs.append(score)
    sortedAUPRCs = np.array(bootstrapAUPRCs)
    sortedAUPRCs.sort()
    lower_ci = sortedAUPRCs[int(0.05*len(sortedAUPRCs))]
    upper_ci = sortedAUPRCs[int(0.95*len(sortedAUPRCs))]
    return lower_ci, upper_ci


def showAllMetrics(y_true, y_preds, optimal_threshold=None):
    #ROC
    auroc = roc_auc_score(y_true, y_preds)
    fpr, tpr, thresh = roc_curve(y_true, y_preds)
    lowerROCCI, upperROCCI = bootStrapModelAUROC(y_true, y_preds)
    #PRC
    auprc = average_precision_score(y_true, y_preds)
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    lowerPRCCI, upperPRCCI = bootStrapModelAUPRC(y_true, y_preds)
    #Other metrics
    if optimal_threshold is None:
        optimal_threshold = findOptimalThreshold(fpr, tpr, thresh)
    y_pred = y_preds > optimal_threshold     
    s, sp, p, f1, a = getMetrics(y_true, y_pred)
    
    print(f"AUROC [95%CI] : {auroc:.4f} [{lowerROCCI:.4f} - {upperROCCI:.4f}]")
    print(f"AUPRC [95%CI] : {auprc:.4f} [{lowerPRCCI:.4f} - {upperPRCCI:.4f}]")
    print(f"Optimal cutoff_threshold = {optimal_threshold:.4f}")
    print(f"Sensitivity: {s:.4f}")
    print(f"Specificity: {sp:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {a:.4f}")
    return fpr, tpr, precision, recall, optimal_threshold

def fitAndShowMetrics(best_model, X_train, X_test, y_train, y_test, trainOrTest = 'train'):
    
    best_model.fit(X_train, y_train)
    
    if trainOrTest =='train':
        y_scores_train = best_model.predict_proba(X_train)
        showAllMetrics(y_train.values, y_scores_train[:,1])
    elif trainOrTest =='test':
        y_scores_test = best_model.predict_proba(X_test)
        showAllMetrics(y_test.values, y_scores_test[:,1])

    return 

def plotROC(model, X_train, X_test, y_train, y_test, title='ROC'):
    colors = ['red', 'orange']

    model.fit(X_train, y_train)
    y_scores_train = model.predict_proba(X_train)
    y_scores_test = model.predict_proba(X_test)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train[:,1])
    fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test[:,1])
    plt.plot(fpr_train, tpr_train, color=colors[0], label='Train (AUROC = %0.4f)' %roc_auc_score(y_train, y_scores_train[:,1]))
    plt.plot(fpr_test, tpr_test, color=colors[1], label='Test (AUROC = %0.4f)' %roc_auc_score(y_test, y_scores_test[:,1]))
    plt.plot([0,1],[0,1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
    return

def plotPRC(model, X_train, X_test, y_train, y_test, title='Precision Recall Curve'):
    colors = ['darkblue', 'blue']

    model.fit(X_train, y_train)
    y_scores_train = model.predict_proba(X_train)
    y_scores_test = model.predict_proba(X_test)
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_scores_train[:,1])
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_scores_test[:,1])
    plt.plot(precision_train, recall_train, color=colors[0], label='Train (AUPRC = %0.4f)' %average_precision_score(y_train, y_scores_train[:,1]))
    plt.plot(precision_test, recall_test, color=colors[1], label='Test (AUPRC = %0.4f)' %average_precision_score(y_test, y_scores_test[:,1]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
    return

def customScaler(X_train, X_test, standard_columns, minMax_columns, X_val=None):
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
    if X_val is not None:
        standard_rescale_val = X_val[standard_columns]
        minmax_rescale_val = X_val[minMax_columns]
        X_val_rescaled = pd.concat([pd.DataFrame(standard.transform(standard_rescale_val), columns=standard_columns), 
                                      pd.DataFrame(minmax.transform(minmax_rescale_val), columns=minMax_columns)], 
                                        axis=1)
        return X_train_rescaled, X_val_rescaled, X_test_rescaled
    
    return X_train_rescaled, X_test_rescaled

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    
    source: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p