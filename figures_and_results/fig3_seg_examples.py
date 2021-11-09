import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
import h5py
import pydicom
import numpy as np
from collections import OrderedDict
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom, median_filter, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from captum.attr import visualization as viz
import sys
sys.path.append('../')
from imaging.segmentation.get_segmentations import get_id2path
L3_SLICE_DIR = '/PATH_TO/l3_slices/'
MODEL_SEG_DIR = '/PATH_TO/l3_slices_8k_segmented/'
ATTRIBUTION_DIR = '/PATH_TO/attributions/'
PREDS_DIR = '/PATH_TO/predictions/'
AbCT_SEG_DIR = '/PATH_TO/abctseg/'




def getDataFromSegmentationH5(segmentation_h5_path):
    with h5py.File(MODEL_SEG_DIR+segmentation_h5_path,'r') as ex:
        base_key = [x for x in ex.keys()][0]
        keys = [x for x in ex[base_key].keys()]
        masks = [ex[base_key][x]["mask"][()] for x in ex[base_key].keys()]
    return keys, masks

def getOrigPathFromSegPath(seg_h5_path):
    dcm_to_look_for = seg_h5_path.replace('.h5','.dcm')
    for root, dirs, names in os.walk(L3_SLICE_DIR):
        if dcm_to_look_for in names:
            orig_path = os.path.join(root, dcm_to_look_for)
            return orig_path
    print("Original path not found")
    return

def bed_removal(image, return_mask=False):
    """
    implementation of automated bed detection and removal from abdominal ct images for automatic segmentation applications
    https://ieeexplore.ieee.org/document/8626638. Note that second stage is not implemented due to only using 1 slice.
    """
    img = median_filter(image, size=5)
    thresh = threshold_otsu(img)
    binary = img > thresh
    all_labels = label(binary)
    label2area = [(x.label, x.area) for x in regionprops(all_labels)]
    label2area.sort(key = lambda x: x[1], reverse=True)
    mask = np.where(all_labels == label2area[0][0], 1 ,0)
    
    if return_mask == True:
        return mask
    
    return np.multiply(image, mask)

def getCTImgBodyMask(dcm_path):
    dcm = pydicom.read_file(dcm_path)
    orig_img = dcm.pixel_array
    body = bed_removal(np.clip(orig_img+dcm.RescaleIntercept,-1000,1000), return_mask=True)
    all_body = binary_fill_holes(body).astype(int)
    
    return np.clip(orig_img+dcm.RescaleIntercept,-1000,1000), all_body

def getFinalTissueTypes(seg_file_path, orig_tissues, orig_masks):
    orig_l3_dcm_path = getOrigPathFromSegPath(seg_file_path)
    orig_img, body = getCTImgBodyMask(orig_l3_dcm_path)

    background = 1-body

    new_tissues = {}

    for i,tm in enumerate(orig_masks):
        if orig_tissues[i] not in ['background','bone']:
            body -= tm.astype(int) 
            new_tissues[orig_tissues[i]] = tm
            
    body[body<0] = 0

    new_tissues['other_tissues'] = body == 1
    new_tissues['background'] = background == 1

    return orig_img, new_tissues

def getImgAttrsByTissue(patient_id, attr_file_path, seg_file_path, tissue_names, tissue_masks):
    
    with h5py.File(attr_file_path,'r') as attr_h5:
        attr = attr_h5[patient_id][()]
        resized_attr = zoom(attr, 512/528, order=1)

    orig_img, new_tissues = getFinalTissueTypes(seg_file_path, tissue_names, tissue_masks)

    return new_tissues, \
            ['orig_img', 'attribution']+[x for x in new_tissues.keys()], \
            [orig_img, resized_attr]+[np.multiply(resized_attr,x) for x in new_tissues.values()]

def getSegmentationAndSaliency(patient_id, attr_file_path, seg_file_path):
    

    seg_keys, seg_masks = getDataFromSegmentationH5(seg_file_path)

    final_masks, labels, attributions = getImgAttrsByTissue(patient_id, attr_file_path, seg_file_path, seg_keys, seg_masks)

    return labels, final_masks, attributions

def load_preds(cohort='5y'):
    if cohort == '1y':
        predictions_fpath = PREDS_DIR + 'IHD_8139_preds_all_1y.csv'
    elif cohort=='5y':
        predictions_fpath = PREDS_DIR + 'IHD_8139_preds_all_5y.csv'
    else:
        raise ValueError('Incorrect cohort specified: i.e. not "1y" or "5y"')

    preds = pd.read_csv(predictions_fpath)
    preds['label_binary'] = preds.apply(lambda x: 'IHD' if x['label']==1 else 'non-IHD', axis=1)

    return preds

def get_saliency_fpath(cohort='5y'):
    if cohort == '1y':
        saliency_vals_path = ATTRIBUTION_DIR + '1y_saliency.h5'
    elif cohort=='5y':
        saliency_vals_path = ATTRIBUTION_DIR + '5y_saliency.h5'
    else:
        raise ValueError('Incorrect cohort specified: i.e. not "1y" or "5y"')
    return saliency_vals_path
def plot_example(pt_id, predictions, saliency_vals_path, id2l3name, show_titles=False, show_color_legend=False, save_path='.figs/sample_segmentations'):
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.5, '#ff0000'),
                                                  (1, '#ff0000')], N=256)
    risks = predictions[predictions['anon_id']==pt_id].to_dict(orient='list')
    seg_file_path = id2l3name[pt_id].replace('.dcm','.h5')
    labels, final_masks, attributions = getSegmentationAndSaliency(pt_id, saliency_vals_path, seg_file_path)

    masked = np.zeros_like(attributions[0])
    colors = ['yellow', 'red', 'gray','orange','green', 'black']

    for i,k in enumerate(['sat','muscle', 'vat', 'other_tissues', 'background']):
        masked[final_masks[k]] = i+1

    fig, ax = plt.subplots(1,3,figsize=(24,16), gridspec_kw = {'wspace':0.01, 'hspace':0})
    ax[0].imshow(np.clip(attributions[0],-50,250), cmap='bone')
    ax[0].get_xaxis().set_visible(False)
    pce_risk = risks['pce_risk'][0]
    fusion_risk = risks["img_clin_fusion_preds"][0]
    label = risks["label_binary"][0]
    ylabel_text = f'{"PCE Risk":^15s}{"Fusion Risk":^15s}{"Outcome":^15s}\n'+\
                    f"{f'{100*pce_risk:.1f}%':^15s}{f'{100*fusion_risk:.1f}%':^15s}{label:^15s}"
    ax[0].set_ylabel(ylabel_text, fontsize=18)
    ax[0].set_yticks([])


    m_img = ax[1].imshow(masked, cmap=ListedColormap(colors))
    ax[1].axis('off')
    
    attributions_vis = attributions[1].copy()
    attributions_vis[attributions[1]<0.1]=0
    
    m_img_f2, m_img_ax2 =  viz.visualize_image_attr(attr=np.dstack([attributions_vis]*3),
                                 original_image=np.dstack([masked/np.max(masked)]*3),
                                 method='blended_heat_map',
                                 show_colorbar=False,
                                 outlier_perc=.8,
                                 plt_fig_axis = (fig,ax[2]),
                                 use_pyplot=False,
                                cmap=default_cmap,
                                sign='absolute_value',
                                 alpha_overlay=.5)

    ax[2].axis('off')
    
    if show_color_legend==True:

        values = [x for x in range(1,1+np.max(masked.astype(int)))]

        colors = [m_img.cmap(m_img.norm(value)) for value in values]
        tissue_names = ['SAT', 'Muscle', 'VAT', 'Other', 'Background']
        patch_labels = [f"{tissue_names[i]}" for i in range(len(values))]
        patches = [mpatches.Patch(edgecolor='none', facecolor=colors[i], label=patch_labels[i]) for i in range(len(values))]

        # create blank rectangle
        extra = mpatches.Rectangle((0, 0), 1,1, fc=None, fill=False, edgecolor='none', linewidth=0)

        #Create organized list containing all handles for table. Extra represent empty space
        legend_handle = patches[:2] + [extra]*2 + patches[2:4] + [extra]*2 + [patches[4], extra] + [extra]*2

        #Define the labels
        label_col_1 = [""]*2
        label_j_1 = tissue_names[:2]
        label_col_3 = [""]*2
        label_j_2 = tissue_names[2:4]
        label_col_5 = [""]*2
        label_j_3 = [tissue_names[-1]]+[""]
        label_empty = [""]

        #organize labels for table construction
        legend_labels = np.concatenate([label_col_1, label_j_1, label_col_3, label_j_2, label_col_5, label_j_3]) #, label_j_1, label_empty * 3, label_j_2, label_empty * 3, label_j_3, label_empty * 3

        ax[1].legend(legend_handle, legend_labels, handlelength=.8,columnspacing=.5,labelspacing=0.05, shadow=False, ncol =6, bbox_to_anchor=(.93, .94), 
                     handletextpad = -.4, loc=0, borderaxespad=-1 , fontsize=24)#shadow = False, handletextpad = -2)
        ax[1].get_legend().set_alpha(0.1)
        
        cbar = fig.colorbar(m_img_ax2.images[-1], ax=ax, orientation='horizontal', shrink=.24, anchor=(.95,5.9), ticks=[0, 1]) 
        cbar.ax.set_xticklabels(['0','>.8'], fontsize=24, horizontalalignment='center')
        cbar.ax.set_xlabel('|saliency|', fontsize=24, labelpad=-15)
    
    if show_titles:
        ax[0].set_title('L3 slice', fontsize=40)
        ax[1].set_title('Segmentation', fontsize=40)
        ax[2].set_title('Tissue Saliency', fontsize=40)
    plt.savefig(save_path, bbox_inches='tight')

if __name__=='__main__':
    low_pce_low_fusion_control = 'Q_gup2trpcO5HMY5pXYIf+tJTRplspO7xZDzVNKWPWs='
    high_pce_high_fusion_case = 'HmoHvTA_ETDF8C4nZosPHxdTlpOFouYTuwilP3IKITk='
    high_pce_low_fusion_control = 'MVSmFlBw8_JKNNZJPC7ngN7dOXe3Bv6jUxl30k+CkrQ='
    low_pce_high_fusion_case = '+QUuphnLmW3oFMe2VLPOmMUIaLqeACTxAIyt8R_LCx4='
    
    id2l3name = get_id2path()
    preds_5y = load_preds('5y')
    saliency_5y_path = get_saliency_fpath('5y')    
    
    plot_example(low_pce_low_fusion_control, preds_5y, saliency_5y_path, id2l3name, True, True, './figs/Fig3_0_low_pce_low_fusion_control.png')
    plot_example(high_pce_high_fusion_case, preds_5y, saliency_5y_path, id2l3name, False, False, './figs/Fig3_1_high_pce_high_fusion_case.png')
    plot_example(high_pce_low_fusion_control, preds_5y, saliency_5y_path, id2l3name, False, False, './figs/Fig3_2_high_pce_low_fusion_control.png')
    plot_example(low_pce_high_fusion_case, preds_5y, saliency_5y_path, id2l3name, False, False, './figs/Fig3_3_low_pce_high_fusion_case.png')