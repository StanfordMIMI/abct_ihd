import os
from pydicom import dcmread
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from fig3_seg_examples import L3_SLICE_DIR, MODEL_SEG_DIR
from numpy import clip
import sys
sys.path.append('../')
from imaging.segmentation.get_segmentations import get_id2path

def getDataFromExample(example_path): 
    with h5py.File(MODEL_SEG_DIR+example_path,'r') as ex:
        base_key = [x for x in ex.keys()][0]
        keys = [x for x in ex[base_key].keys()]
        masks = [ex[base_key][x]["mask"][()] for x in ex[base_key].keys()]
        csa = [ex[base_key][x]["Cross-sectional Area (mm^2)"][()] for x in ex[base_key].keys()]
        avg_hu = [ex[base_key][x]["Hounsfield Unit"][()] for x in ex[base_key].keys()]
    return keys, masks, csa, avg_hu

def main():
    
    img_ID = '87z_HBbw_0pksYCzkwXkWsxa615kMqJkldv_9sY7B0I='
    id2l3name = get_id2path()
    img_path = id2l3name[img_ID]
    
    #l3 slice
    l3_img = dcmread(L3_SLICE_DIR+img_path)
    plt.figure(figsize=(24,24))
    plt.imshow(clip(l3_img.pixel_array + l3_img.RescaleIntercept, -50,250), cmap='bone')
    plt.axis('off')
    plt.savefig('./figs/fig_1_l3_slice.png', bbox_inches='tight')

    
    names, masks, csa, avg_hu = getDataFromExample(img_path.replace('.dcm','.h5'))
    plt.figure(figsize=(24,24))
    
    
    colors_m = [(0,0,0,0), 'red']
    colors_sat = [(0,0,0,0), 'yellow']
    colors_vat = [(0,0,0,0), 'orange']

    plt.figure(figsize=(12,12))
    plt.imshow(clip(l3_img.pixel_array+l3_img.RescaleIntercept, -50, 250), cmap='bone')
    plt.imshow(masks[2], cmap=ListedColormap(colors_m),alpha=1)
    plt.imshow(masks[3], cmap=ListedColormap(colors_sat),alpha=1)
    plt.imshow(masks[4], cmap=ListedColormap(colors_vat),alpha=1)

    plt.axis('off')
    plt.savefig('./figs/fig_1_l3_slice_segmented.png', bbox_inches='tight')

    plt.imshow
if __name__=='__main__':
    main()