"""
Get and Plot Observed/Expected Tissue Saliency and P-values
"""
import os
import pydicom
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset_dataloader import bed_removal
from scipy.ndimage import zoom, binary_fill_holes
from scipy.stats import ttest_rel

def getID2Path():
    """
    Returns dictionary mapping an image ID to its l3_slice path
    """
    id2path = pd.read_csv('/PATH_TO/data/l3_slices_path.csv')
    id2path['filename'] = id2path.apply(lambda x: os.path.basename(x['l3_slice_path']).replace('.dcm','.h5'), axis=1)
    id2path = id2path[['filename','id']].set_index('id').to_dict()['filename']
    return id2path

def getOrigImagePath(h5_segmentation_path):
    l3_path = '/PATH_TO/data/l3_slices/'
    expected_dcm_name = h5_segmentation_path.replace('.h5','.dcm')
    for root, _, names in os.walk(l3_path):
        if expected_dcm_name in names:
            orig_path = os.path.join(root, expected_dcm_name)
            break
    return orig_path
def getOtherTissueMask(orig_dicom_path, orig_tissue_names, orig_tissue_masks):
    """
    Creates mask for other tisssues (missing from seg model) by creating body mask and subtracting
    muscle, VAT and SAT
    """
    dcm = pydicom.read_file(orig_dicom_path)
    orig_img = dcm.pixel_array

    body = bed_removal(np.clip(orig_img+dcm.RescaleIntercept,-1000,1000), return_mask=True)
    other_tissues = binary_fill_holes(body).astype(int)
    background = 1-other_tissues

    new_tissues = {}
    for i,tm in enumerate(orig_tissue_masks):
        if orig_tissue_names[i] not in ['background','bone']:
            other_tissues -= tm.astype(int)
            new_tissues[orig_tissue_names[i]] = tm
        other_tissues[other_tissues<0] = 0
    new_tissues['other_tissues'] = other_tissues
    new_tissues['background'] = background

    return [x for x in new_tissues.keys()], [x for x in new_tissues.values()]

def getDataFromSegmentationH5(h5_segmentation_path):
    """
    Returns segmentation names and masks from output file
    """
    out_dir = '/PATH_TO/data/segmented_l3_slices/'
    with h5py.File(out_dir+h5_segmentation_path,'r') as ex:
        base_key = [x for x in ex.keys()][0]
        keys = [x for x in ex[base_key].keys()]
        masks = [ex[base_key][x]["mask"][()] for x in ex[base_key].keys()]
    modified_keys, modified_masks = getOtherTissueMask(getOrigImagePath(h5_segmentation_path), keys, masks)
    return modified_keys, modified_masks

def getAttributionsByTissue(attributions, tissue_names, tissues):
    """
    Returns Tissue Attributions (saliency) for a list of tissues
    """
    resized_attr = zoom(attributions, 512/528, order=1)
    total_attr = np.sum(np.abs(resized_attr))
    tissue_attrs = {}
    tissue_attrs['total'] = total_attr
    for i, tn in enumerate(tissue_names):
        tissue_attrs[tn] = np.sum(np.multiply(np.abs(resized_attr), tissues[i]))/total_attr
        tissue_attrs['expected_'+tn] =  np.sum(tissues[i])/(512*512)
        
    return tissue_attrs

def getAllAttrsByTissue(attr_file_path, id2path):
    with h5py.File(attr_file_path, 'r') as attr_h5:
        keys = [x for x in attr_h5.keys()]
        attrs = {}

        for k in keys:
            attr = attr_h5[k][()]
            tissue_names, tissue_masks = getDataFromSegmentationH5(id2path[k])
            tissue_attr = getAttributionsByTissue(attr, tissue_names, tissue_masks)
            attrs[k] = tissue_attr

    return attrs

def plotOESaliency(df, subset='test'):
    df = df[df['set']==subset]
    
    _, ax = plt.subplots(1,5, figsize=(24,4), gridspec_kw = {'wspace':0.05, 'hspace':0})

    ax[0].scatter(df['expected_muscle'], df['muscle'], color='red', alpha=0.5) 
    ax[1].scatter(df['expected_sat'], df['sat'], color='yellow', alpha=0.5) 
    ax[2].scatter(df['expected_vat'], df['vat'], color='orange', alpha=0.5)  
    ax[3].scatter(df['expected_other_tissues'], df['other_tissues'], color='green', alpha=0.5) 
    ax[4].scatter(df['expected_background'], df['background'], color='black', alpha=0.5)  


    names = ['Muscle', 'SAT', 'VAT', 'Other', 'Background']
    observed_means = [x for x in df.mean()[['muscle','sat','vat','other_tissues','background']]]
    expected_means = [x for x in df.mean()[['expected_muscle','expected_sat','expected_vat','expected_other_tissues','expected_background']]]

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
    plt.show()

    return

def getObservedExpectedPVals(df, subset='test'):
    p_vals = {}
    for tissue in ['muscle','vat','sat','other_tissues','background']:
        p_vals[tissue] = [None]
        x = df[df['set']==subset]['expected_'+tissue]
        y = df[df['set']==subset][tissue]
        _, p_vals[tissue][0] = ttest_rel(a=x, b=y)

    return p_vals

def main():
    id2path = getID2Path()

    # 1y Tissue saliency
    saliency_vals_path_1y = '/PATH_TO/data/1y_saliency.h5'
    tissue_saliency_path_1y = '/PATH_TO/data/1y_tissue_saliency.csv'

    if os.path.exists(tissue_saliency_path_1y):
        slncy_attrs_df_1y = pd.read_csv(tissue_saliency_path_1y)
    else:
        slncy_attrs_1y = getAllAttrsByTissue(saliency_vals_path_1y, id2path)
        slncy_attrs_df_1y = pd.DataFrame.from_dict(slncy_attrs_1y, orient='index').reset_index().rename(columns={'index':'id'})
        slncy_attrs_df_1y.to_csv(tissue_saliency_path_1y, index=False)

    # 5y Tissue saliency
    saliency_vals_path_5y = '/PATH_TO/data/5y_saliency.h5'
    tissue_saliency_path_5y = '/PATH_TO/data/5y_tissue_saliency.csv'

    if os.path.exists(tissue_saliency_path_5y):
        slncy_attrs_df_5y = pd.read_csv(tissue_saliency_path_5y)
    else:
        slncy_attrs_5y = getAllAttrsByTissue(saliency_vals_path_5y, id2path)
        slncy_attrs_df_5y = pd.DataFrame.from_dict(slncy_attrs_5y, orient='index').reset_index().rename(columns={'index':'id'})
        slncy_attrs_df_5y.to_csv(tissue_saliency_path_5y, index=False)

    #Plot values
    plotOESaliency(slncy_attrs_df_1y)
    plotOESaliency(slncy_attrs_df_5y)

    #Compare paired tissue saliencies
    p_vals_1y = getObservedExpectedPVals(slncy_attrs_df_1y)
    p_vals_5y = getObservedExpectedPVals(slncy_attrs_df_5y)
    print(f'1y Tissue saliency O vs E p-vals: {p_vals_1y}')
    print(f'5y Tissue saliency O vs E p-vals: {p_vals_5y}')

if __name__=='__main__':
    main()
