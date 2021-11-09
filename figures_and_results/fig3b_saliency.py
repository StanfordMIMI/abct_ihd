import os
import pandas as pd
import h5py
from fig3_seg_examples import getOrigPathFromSegPath, getFinalTissueTypes
import sys
sys.path.append('../')
from imaging.segmentation.get_segmentations import get_id2path
from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel

L3_SLICE_DIR = '/PATH_TO/l3_slices/'
SEG_IMG_DIR = '/PATH_TO/l3_slices_8k_segmented/'
SALIENCY_DIR = '/PATH_TO/attributions/'
DATA_DIR = '/PATH_TO/data/'
    
def getDataFromSegmentationH5(h5_segmentation_path):
    with h5py.File(SEG_IMG_DIR+h5_segmentation_path,'r') as ex:
        base_key = [x for x in ex.keys()][0]
        keys = [x for x in ex[base_key].keys()]
        masks = [ex[base_key][x]["mask"][()] for x in ex[base_key].keys()]
    orig_img, seg_dict = getFinalTissueTypes(h5_segmentation_path, keys, masks)
    modified_keys = list(seg_dict.keys())
    modified_masks = [seg_dict[k] for k in modified_keys]
    return modified_keys, modified_masks

def getAttributionsFor1ImgByTissue(attributions, tissue_names, tissues):
    
    resized_attr = zoom(attributions, 512/528, order=1)
    total_attr = np.sum(np.abs(resized_attr))
    tissue_attrs = {}
    tissue_attrs['total'] = total_attr
    for i, tn in enumerate(tissue_names):
        tissue_attrs[tn] = np.sum(np.multiply(np.abs(resized_attr), tissues[i]))/total_attr
        tissue_attrs['expected_'+tn] =  np.sum(tissues[i])/(512*512)
        
    return tissue_attrs

def getAttrsByTissue(attr_file_path, id2path):
    with h5py.File(attr_file_path, 'r') as attr_h5:
        keys = [x for x in attr_h5.keys()]
        attrs = {}

        for k in keys:
            attr = attr_h5[k][()]
            tissue_names, tissue_masks = getDataFromSegmentationH5(id2path[k])
            tissue_attr = getAttributionsFor1ImgByTissue(attr, tissue_names, tissue_masks)
            attrs[k] = tissue_attr

    return attrs

def plotOESaliency(df, subset='test', savepath='./figs/OESaliency.jpeg'):
    df = df[df['set']==subset]
    
    fig,ax = plt.subplots(1,5, figsize=(24,4), gridspec_kw = {'wspace':0.05, 'hspace':0})

    ax[0].scatter(df['expected_muscle'], df['muscle'], color='red', alpha=0.5) 
    ax[1].scatter(df['expected_sat'], df['sat'], color='yellow', alpha=0.5) 
    ax[2].scatter(df['expected_vat'], df['vat'], color='orange', alpha=0.5)  
    ax[3].scatter(df['expected_other_tissues'], df['other_tissues'], color='green', alpha=0.5) 
    ax[4].scatter(df['expected_background'], df['background'], color='black', alpha=0.5)  


    names = ['Muscle', 'SAT', 'VAT', 'Other', 'Background']
    observed_means = [x for x in df.mean()[['muscle','sat','vat','other_tissues','background']]]
    expected_means = [x for x in df.mean()[['expected_muscle','expected_sat','expected_vat','expected_other_tissues','expected_background']]]
    print(f"{[f'{observed_means[i]/expected_means[i]:.2f}' for i in range(len(observed_means))]}")
    print(f" {(sum([observed_means[0], observed_means[2]]))*100:.1f}/ {(sum([expected_means[0], expected_means[2]]))*100:.1f}")
    observed_means_t = [f'{x:.2f}' for x in observed_means]
    expected_means_t = [f'{x:.2f}' for x in expected_means]

    means = [r'$\frac{\bar{%s}}{\bar{%s}}=\frac{%s}{%s}=%.2f$' %('O',
                                                        'E',
                                                        o, 
                                                        expected_means_t[i], float(o)/float(expected_means_t[i])) 
                                                        for i,o in enumerate(observed_means_t)]

    for i in range(5):
        if i==0:
            ax[i].set_ylabel('Observed', fontsize=22)
            ax[i].set_ylim((0,1))
            ax[i].set_yticks([0,0.2,0.4,0.6,0.8,1])
            ax[i].set_yticklabels(labels=['0', '.2', '.4', '.6', '.8', '1'], fontsize=18)
        else:
            ax[i].set_yticks([])
            ax[i].set_ylim((0,1))

        ax[i].set_title(names[i]+' Saliency', fontsize=24)
        ax[i].plot([0,1],[0,1], color='black', linestyle='--')
        ax[i].set_xlabel('Expected', fontsize=22)
        ax[i].set_xlim((0,1))
        ax[i].set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax[i].set_xticklabels(labels=['0', '.2', '.4', '.6', '.8', '1'], fontsize=16)

        # Display averages
        props = dict(boxstyle=None, facecolor='w', edgecolor='w', alpha=0.5)

        # place a text box in upper left in axes coords
        ax[i].text(0.1, .8, means[i], transform=ax[i].transAxes, fontsize=20,
                verticalalignment='top', bbox=props)
    plt.savefig(savepath, bbox_inches='tight')
    
    return 

def getObservedExpectedPVals(df, subset='test'):
    p_vals = {}
    for tissue in ['muscle','vat','sat','other_tissues','background']:
        p_vals[tissue] = [None, None]
        x = df[df['set']==subset]['expected_'+tissue]
        y = df[df['set']==subset][tissue]
        _, p_vals[tissue][0] = wilcoxon(x=x, y=y)
        _, p_vals[tissue][1] = ttest_rel(a=x, b=y)

    return p_vals

def main():
    id2l3name = {k:v.replace('.dcm','.h5') for k,v in get_id2path().items()}
    saliency_attr_path_1y = SALIENCY_DIR + '/saliency_attr_1y.csv'
    if os.path.exists(saliency_attr_path_1y):
        slncy_attrs_df_1y = pd.read_csv(saliency_attr_path_1y)
    else:
        saliency_vals_path_1y = SALIENCY_DIR + '1y_saliency.h5'
        slncy_attrs_1y = getAttrsByTissue(saliency_vals_path_1y, id2l3name)
        slncy_attrs_df_1y = pd.DataFrame.from_dict(slncy_attrs_1y, orient='index').reset_index().rename(columns={'index':'id'})
        slncy_attrs_df_1y.to_csv(saliency_attr_path_1y, index=False)

    slncy_attrs_df_1y = slncy_attrs_df_1y.merge(pd.read_csv(DATA_DIR+'IHD_8139_1y_train_val_test_split.csv')[['anon_id','set']], left_on='id', right_on='anon_id', how='left')

    saliency_attr_path_5y = SALIENCY_DIR + '/saliency_attr_5y.csv'
    if os.path.exists(saliency_attr_path_5y):
        slncy_attrs_df_5y = pd.read_csv(saliency_attr_path_5y)
    else:
        saliency_vals_path_5y = SALIENCY_DIR + '5y_saliency.h5'
        slncy_attrs_5y = getAttrsByTissue(saliency_vals_path_5y, id2l3name)
        slncy_attrs_df_5y = pd.DataFrame.from_dict(slncy_attrs_5y, orient='index').reset_index().rename(columns={'index':'id'})
        slncy_attrs_df_5y.to_csv(saliency_attr_path_5y, index=False)
    
    slncy_attrs_df_5y = slncy_attrs_df_5y.merge(pd.read_csv(DATA_DIR+'IHD_8139_5y_train_val_test_split.csv')[['anon_id','set']], left_on='id', right_on='anon_id', how='left')

    plotOESaliency(slncy_attrs_df_1y, savepath='./figs/fig3b_OESaliency_1y_test.jpeg')
    plotOESaliency(slncy_attrs_df_5y, savepath='./figs/fig3b_OESaliency_5y_test.jpeg')

    p_vals_1y = getObservedExpectedPVals(slncy_attrs_df_1y)
    p_vals_5y = getObservedExpectedPVals(slncy_attrs_df_5y)

    print(p_vals_1y)
    print(p_vals_5y)

if __name__ == '__main__':
    main()