import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from torch.utils.data.dataset import Subset
from scipy.ndimage import zoom
import logging
import pydicom 
import os
import warnings
### Dataset utils
def convert_to_HU(img, rescaleSlope, rescaleIntercept):
    """
    Converts raw pixel values to HU using the slope and intercept from dicom
    """
    hu_img = np.copy(img).astype(np.int16)
    #Set pixels outside of scan to 0
    hu_img[hu_img==-2000] = 0
    
    if rescaleSlope != 1:
        hu_img = rescaleSlope * hu_img.astype(np.float64)
        hu_img = hu_img.astype(np.int16)
    
    hu_img += np.int16(rescaleIntercept)
    
    return np.array(hu_img, dtype=np.int16)

def load_data(data_path, keep_all_keys=True, dataset_path=None, label_mapper=None):
    """
    Loads images, spacings, labels (outcomes) and ids (names)
    Params: 
    -data_path: path to H5 file containing images
    -keep_all_keys: whether to keep all keys from h5 or search a specific subset of IDs
    -dataset_path: path to .csv containing ids of interest
    -label_mapper: map label in .csv to 0 or 1 (nonIHD or IHD)
    """

    with h5.File(data_path, 'r') as h5f:
        if keep_all_keys==True:
            names = [x for x in h5f.keys()]
        else:
            keys_to_keep = getIDsofInterest(dataset_path)
            names = [x for x in h5f.keys() if x in keys_to_keep]

        if keep_all_keys==True:
            cohort2_labels = [h5f[k].attrs['c2_label'] for k in names]
            _, c2_labels = np.unique(cohort2_labels, return_inverse=True)
            labels = np.zeros((c2_labels.shape[0],2))
            labels[range(c2_labels.shape[0]), c2_labels] = 1
        else:
            labels = getLabels(names, dataset_path, label_mapper)

        pixel_spacings = [h5f[k].attrs['pixel_spacing'] for k in names]
        thicknesses = np.asarray([h5f[k].attrs['slice_thickness'] for k in names])
        spacings, thicknesses = mergePixelSpacingAndThickness(pixel_spacings, thicknesses)

        images = []
        for _,k in enumerate(names):
            images.append(h5f[k][()])

    return images, spacings, labels, names


def normalize_and_preprocess(images, num_neighbors, val_zero, val_one, spacings, normalize_spacing=False,\
                            new_spacing=1, final_size=(3,256,256), mean_fat=False, remove_bed=False,\
                            as_integer=False):
    """
    Returns a preprocessed np array containing images
    Params:
        images: original numpy array containing images with pixels in HU
        num_neighbors: number of slices to take above/below L3 level (0 for 2D, 2 for 2.5D)
        val_zero: HU below which pixels will be set to black (0)
        val_one: HU above which pixel values will be set to white (1 or 255)
        spacings: list containing [x,y,z] pixel spacings for each image
        normalize_spacing: whether or not to normalize spacing
        new_spacing: new spacing in mm to normalize image to
        final_size: shape of final image e.g. 3,256,256
        mean_fat: whether or not to modify HU in [-190,-30] to -110
        remove_bed: whether or not to remove bed from field of view
        as_integer: whether or not to represent as integer
    Returns:
        images_norms: array of normalized images
    """
    images_norm = []

    for image, s in zip(images, spacings): 
        input_shape = image.shape
        mid = int(input_shape[2]/2)
        if num_neighbors == 0:
            image = image[:,:,mid]
            zeroToOne = normalizeOneSlice(image, mean_fat, s, normalize_spacing, new_spacing,\
                                        final_size, val_zero, val_one, remove_bed, as_integer)
            stacked = np.repeat(zeroToOne[:,:,np.newaxis], 3, axis=2)
        else:
            raise NotImplementedError('Only 1 slice supported.')
        images_norm.append(stacked)

    return np.array(images_norm)
def normalize_and_preprocess_dcm(images, val_zero, val_one, spacings, normalize_spacing=False,\
                            new_spacing=1, final_size=(3,256,256), mean_fat=False, remove_bed=False,\
                            as_integer=False):
    """
    Returns a preprocessed np array containing images
    Params:
        images: original numpy array containing images with pixels in HU
        val_zero: HU below which pixels will be set to black (0)
        val_one: HU above which pixel values will be set to white (1 or 255)
        spacings: list containing [x,y,z] pixel spacings for each image
        normalize_spacing: whether or not to normalize spacing
        new_spacing: new spacing in mm to normalize image to
        final_size: shape of final image e.g. 3,256,256
        mean_fat: whether or not to modify HU in [-190,-30] to -110
        remove_bed: whether or not to remove bed from field of view
        as_integer: whether or not to represent as integer
    Returns:
        images_norms: array of normalized images
    """
    images_norm = []
    import matplotlib.pyplot as plt
    for image, s in zip(images, spacings): 
        input_shape = image.shape    
        zeroToOne = normalizeOneSlice(image, mean_fat, s, normalize_spacing, new_spacing,\
                                    final_size, val_zero, val_one, remove_bed, as_integer)
        stacked = np.repeat(zeroToOne[:,:,np.newaxis], 3, axis=2)
        images_norm.append(stacked)
    return images_norm
def normalizeOneSlice(image, mean_fat, orig_spacing, normalize_spacing, new_spacing,\
                        final_size, val_zero, val_one, remove_bed, as_integer):
    """
    Normalizes a CT slice into a 0-1 Float (or 0-255 Int) of specified desired dimensions
    """
    if mean_fat:
        image = normalize_fat(image)
    if normalize_spacing:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = zoom(image, [new_spacing/x for x in orig_spacing[:2]]) 
    masked = np.clip(image, val_zero, val_one)
    orig_size = masked.shape
    resized = zoom(masked, [final_size[1]/orig_size[0], final_size[2]/orig_size[1]])
    zeroToOne = normalize_to_zero_one(resized, val_zero, val_one)
    if remove_bed:
        zeroToOne = bed_removal(zeroToOne)
        
    if as_integer:
        zeroToOne *= 255
        zeroToOne = zeroToOne.astype('uint8')
        
    return zeroToOne

def getIDsofInterest(split_path,label_col):
    """
    Returns IDs that are found in a dataset 
    Params:
        -split_path: a .csv file that includes the train/val/test split for each ID
    Returns:
        -ids: set of IDs (those found in the split_path file)
    """
    data = pd.read_csv(split_path)
    return set(data[~pd.isna(data[label_col])]['anon_id'])

def getLabels(ids, dataset_path, label_col):
    """
    Returns labels for IDs in a dataset 
    Params:
        -ids: list of ids to return labels for. labels are returned in order of ids.
        -split_path: a .csv file that includes the label for each ID
        -label_col: string containing binary label (0/1)
    Returns:
        -labels: np array of shape (len(ids), 2) with a 1 in the 0-th or 1-st column indicating negative/positive label
    """
    data = pd.read_csv(dataset_path).set_index('anon_id')
    data = data[~pd.isna(data[label_col])]

    id2label = {k:int(v) for k,v in data.to_dict()[label_col].items()}
    
    labels = np.zeros((len(ids),2))
    
    for i,ID in enumerate(ids):
        labels[i,id2label[ID]] = 1
    assert np.sum(labels) == len(ids)
    return labels

class dcmDataset(torch.utils.data.Dataset):
    def __init__(self, config, dcm_dir, transform=None):
        self.config = config
        self.dcm_dir = dcm_dir
        self.transform = None
        self.raw_data = self.load_dicoms()
        self.keys = [x['ID'] for x in self.raw_data]
        

        images = normalize_and_preprocess_dcm(images=[x['img'] for x in self.raw_data], 
                                                val_zero=self.config['thresh_low'],
                                                val_one=self.config['thresh_high'], 
                                                spacings=[x['spacing'] for x in self.raw_data], 
                                                normalize_spacing=self.config['normalize_spacing'], 
                                                new_spacing=self.config['new_spacing'], 
                                                final_size=self.config['new_img_size'], 
                                                mean_fat=self.config['normalize_fat'], 
                                                remove_bed=self.config['remove_bed'], 
                                                as_integer=self.config['as_integer'])
                                              
        self.images = {k:v for k,v in zip(self.keys, images)}
        # self.images={x['ID']:x['img'] for x in self.raw_data}                                              
    def load_dicoms(self):
        data = []
        dcm_files = os.listdir(self.dcm_dir)
        for f in dcm_files:
            dcm = pydicom.dcmread(os.path.join(self.dcm_dir, f))
            pixels = convert_to_HU(dcm.pixel_array, dcm.RescaleSlope, dcm.RescaleIntercept)
            spacing = dcm.PixelSpacing
            thickness = dcm.SliceThickness
            data.append({'ID':f.replace('.dcm',''), 'img':pixels, 'spacing':spacing, 'thickness':thickness})
        return data
        
    def setTransform(self, transform):
        self.transform = transform
        return
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):        
        k = self.keys[index]
        x = self.images[k]
        if self.transform is not None:
            x = self.transform(x)
        return x, k

class h5Dataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.fpath = self.config['data_file']
        self.transform = transform
        if 'cache_path' in self.config:
            self.cache_path = self.config['cache_path']
        else:
            self.cache_path = None

        if '1y' in self.config['split_file']:
            self.cohort_label_name='1y_label'
        elif '5y' in self.config['split_file']:
            self.cohort_label_name='5y_label'
        else:
            raise ValueError('Incorrect cohort split file specified (must contain 1y or 5y in name of file)')
        self.check_cache()
        
        self.preload_all_data = config['preload_all_data']
        if self.preload_all_data:
            with np.load(self.cache_path) as data:
                self.data = {}
                self.data['names'] = data['names']
                self.data['images'] = data['images']
                self.data['labels'] = data['labels']
                self.keys = self.data['names']
                self.length = len(self.keys)
    def __getitem__(self, index):        
        k = self.keys[index]
            
        if self.cache_path is not None:
            if self.preload_all_data:
                x = self.data['images'][index]
                y = self.data['labels'][index]
            else:
                with np.load(self.cache_path) as data:
                    x = data['images'][index]
                    y = data['labels'][index]

        else:
            with h5.File(self.fpath, 'r') as h5f:
                x = h5f[k][()]
                y = h5f[k].attrs[self.cohort_label_name]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, k
    
    def __len__(self):
        return self.length
    
    def setTransform(self, transform):
        self.transform = transform
        return
    
    def check_cache(self):
        cache_filename = os.path.basename(self.fpath).split('.')[0]
        
        if 'cache_path' not in self.config.keys():
            self.config['cache_path'] = self.config['model_path']
            
        cache_path = os.path.join(self.config['cache_path'], cache_filename + '_cache.npz')

        if self.config['use_cache'] and os.path.exists(cache_path):
            logging.info(f'using cache from {cache_path}')
        else:
            images, spacings, labels, names = load_data(self.fpath, dataset_path=self.config['split_file'], \
                                                           cohort_label=self.cohort_label_name)
            if 'bychannel' in self.config: 
                logging.info("about to preprocess by channel")
                logging.info(self.config['bychannel'])
                images = normalize_and_preprocess_bychannel(images, self.config['thresh_low'], \
                                              self.config['thresh_high'], spacings, self.config['normalize_spacing'], \
                                              self.config['new_spacing'], self.config['new_img_size'], \
                                              self.config['remove_bed'], self.config['as_integer'],
                                            mode=self.config['bychannel'])
            else:
                images = normalize_and_preprocess(images, self.config['thresh_low'], \
                                              self.config['thresh_high'], spacings, self.config['normalize_spacing'], \
                                              self.config['new_spacing'], self.config['new_img_size'], self.config['normalize_fat'], \
                                              self.config['remove_bed'], self.config['as_integer'])
            args = {'number neighboring slices':0, 
                    'clipping values': (self.config['thresh_low'], self.config['thresh_high']),
                    'spacing normalized': self.config['normalize_spacing'], 
                    'new spacing (if normalized)': self.config['new_spacing'], 
                    'new image size':self.config['new_img_size'],
                    'values in fat HU range normalized to the mid-value':self.config['normalize_fat'],
                   'bed removed with traditional image processing':self.config['remove_bed'],
                   'pixel values converted to uint8': self.config['as_integer']}
            np.savez(cache_path, images=images, spacings=spacings,
                                labels=labels, names=names, params=args)
        self.cache_path = cache_path
        return





### Dataloader utils
def normalize_fat(img, fat_low=-190, fat_high=-30):
    """
    Sets all values within a range to the mean of that range (usually fat HU values)
    """
    img[(img >= fat_low) & (img <= fat_high)] = (fat_low+fat_high)/2
    return img

def normalize_to_zero_one(image, minv=-1000, maxv=1000, eps=1e-7):
    """
    preprocesses image by:
        -normalizing image into float 0 to 1
    """
    normalized = image - minv
    normalized /= (maxv-minv+eps)

    return normalized

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

def mergePixelSpacingAndThickness(pixel_spacings, thicknesses):
    spacings = np.empty((pixel_spacings.shape[0], pixel_spacings.shape[1]+1))
    spacings[:,:2] = pixel_spacings
    spacings[:,2] = thicknesses
    return spacings, thicknesses

def split_by_idx(dataset, indices):
    """
    Split a dataset into non-overlapping new datasets with each given index.

    Arguments:
        dataset (dataset): Dataset to be split
        lengths (indices): list of lists of indices for each split
    """

    if sum([len(x) for x in indices]) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    return [Subset(dataset, idxs) for idxs in indices]

def trainValTestSplit(dataset, split_path):
    """
    Return train/val/test splits from a dataset according to specifications in split_path
    
    """

    split_data = pd.read_csv(split_path)
    split_dict = split_data.set_index('id').to_dict()['set']
    train_idx = []
    val_idx = []
    test_idx = []
    for i,x in enumerate(dataset.data['names']):
        split = split_dict[x]
        if split == 'train':
            train_idx.append(i)
        elif split == 'val':
            val_idx.append(i)
        elif split == 'test':
            test_idx.append(i)
        else:
            print(split)
    train_split, val_split, test_split = split_by_idx(dataset, [train_idx, val_idx, test_idx])

    return train_split, val_split, test_split
