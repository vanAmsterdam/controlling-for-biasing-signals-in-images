import os
import numpy as np
import pandas as pd
import pylidc as pl
import pydicom
import feather # for writing data frame to disk (works with R)
import pickle
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer

def flatten_annotation(ann):
    '''
    Flattens annotations into a single row that can be added to a pandas DataFrame
    '''
    
    id_vals = np.array([
        ann.scan.patient_id,
        ann._nodule_id,
        ann.id,
        ann.scan_id], 
        dtype = '<U14')
    feature_vals = ann.feature_vals()
    size_vals = np.array([ann.volume, ann.surface_area, ann.diameter])
    return(id_vals, feature_vals, size_vals)

def annotation_to_dict(ann):
    '''
    Flattens annotations into a single row pandas DataFrame
    '''
    
    d = {
        'patient_id': ann.scan.patient_id,
        'nodule_id_lidc': ann._nodule_id,
        'annotation_id_lidc': ann.id,
        'scan_id': ann.scan_id,
        'volume': ann.volume,
        'surface_area': ann.surface_area,
        'diameter': ann.diameter
    }
    feature_names = ['sublety', 'internalstructure', 'calcification',
               'sphericity', 'margin', 'lobulation', 'spiculation',
               'texture', 'malignancy']
    for feature, value in zip(feature_names, ann.feature_vals()):
        d[feature] = value
    return d

def annotation_to_df(ann):
    try:
        d = annotation_to_dict(ann)
        d = {k: [v] for k, v in d.items()}
        df = pd.DataFrame.from_dict(d)
    except:
        df = None
    return df

def annotation_list_to_df(anns):
    assert isinstance(anns, list)
    dfs = []
    for ann in anns:
        dfs.append(annotation_to_df(ann))
    df = pd.concat(dfs, ignore_index=True)
    df["annotation_idx"] = range(1, df.shape[0]+1)
    return df

def flatten_annotations(annotations):
    '''
    Take a list of annotations, return a pandas DataFrame
    '''
    if not isinstance(annotations, list):
        # makes sure that anns is a list, even if it is of length 1
        annotations = [annotations]

    # instantiate empty arrays for the values
    id_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[0].shape[0]), dtype = "<U14")
    feature_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[1].shape[0]), dtype = "int64")
    size_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[2].shape[0]), dtype = np.float32)
    
    # loop over list of annotations
    for i, ann in enumerate(annotations):
        id_vals, feature_vals, size_vals = flatten_annotation(ann)
        id_values[i,:] = id_vals
        feature_values[i,:] = feature_vals
        size_values[i, :] = size_vals
    
    # combine together in a pandas DataFrame
    df_ids  = pd.DataFrame(id_values, columns = ["patient_id", "nodule_id", "annotation_id", "scan_id"])
    df_feat = pd.DataFrame(feature_values, columns = [
                                         'sublety', 'internalstructure', 'calcification',
                                         'sphericity', 'margin', 'lobulation', 'spiculation',
                                         'texture', 'malignancy'])
    df_size = pd.DataFrame(size_values, columns = ["volume", "surface_area", "diameter"])
    df = pd.concat([df_ids, df_feat, df_size], axis = 1)
    return(df)


def flatten_annotations_by_nodule_cluster(scans, dir):
    '''
    take a list of scans, return a pandas DataFrame
    '''
    
    # instantiate DataFrame
    df = flatten_annotations(scans[0].annotations[0]).iloc[0:0]
    df.assign(nodule_number = np.empty(0, dtype = "int32"))
    
    # loop over scans
    print("converting data into pandas dataframe")
    for scan in tqdm(scans):
        patient_id = scan.patient_id[-4:]
        print("converting data into pandas dataframe for patient {}".format(patient_id))
        with open(os.path.join(dir, "nodule-clusters", patient_id+".pkl"), "rb") as f:
            nodule_annotations = pickle.load(f)
        # loop over nodules within a scan
        for i, nodule_annotations in enumerate(nodule_annotations):
            if not isinstance(nodule_annotations, list):
                # makes sure that anns is a list, even if it is of length 1
                nodule_annotations = [nodule_annotations]
            nodule_df = flatten_annotations(nodule_annotations)
            nodule_df = nodule_df.assign(nodule_number = i+1)
            df = pd.concat([df, nodule_df], axis = 0)
    return(df)

def flatten_annotations_by_nodule(scans):
    '''
    take a list of scans, return a pandas DataFrame
    '''
    
    # instantiate DataFrame
    df = flatten_annotations(scans[0].annotations[0]).iloc[0:0]
    df.assign(nodule_number = np.empty(0, dtype = "int32"))
    
    # loop over scans
    for scan in scans:
        # loop over nodules within a scan
        for i, nodule_annotations in enumerate(scan.cluster_annotations()):
            if not isinstance(nodule_annotations, list):
                # makes sure that anns is a list, even if it is of length 1
                nodule_annotations = [nodule_annotations]
            nodule_df = flatten_annotations(nodule_annotations)
            nodule_df = nodule_df.assign(nodule_number = i+1)
            df = pd.concat([df, nodule_df], axis = 0)
    return(df)

def get_intercept_and_slope(scan):
    ''' 
    scan is the results of a pydicom query
    returns the intercept and slope
    adapted from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    dcm_path = scan.get_path_to_dicom_files()
    dcm_files = [x for x in os.listdir(dcm_path) if x.endswith(".dcm")]
    slice0 = pydicom.read_file(os.path.join(dcm_path, dcm_files[0]), stop_before_pixels=True)
    intercept = slice0.RescaleIntercept
    slope = slice0.RescaleSlope
    return intercept, slope

def crop_nodule_tight_z(ann, volume=None, scan=None, scan_spacing=None, out_size_cm = 5):
    """
    Get nodule cropped tightly in z direction, but of minimum dimension in xy plane
    """
    # print(f"trying to crop")
    if volume is None:
        if scan is None:
            scan = ann.scan
        volume = scan.to_volume()
    if scan_spacing is None:
        scan_spacing = scan.pixel_spacing
    # print(f"scan_spacing: {scan_spacing}")
    padding = get_padding_tight_z(ann, scan_spacing=scan_spacing, out_size_cm=out_size_cm)
    # print(f"padding: {padding}")

    mask = ann.boolean_mask(pad=padding)
    bbox = ann.bbox(pad=padding)
    zvals= ann.contour_slice_indices
    arr  = volume[bbox]

    return arr, mask, zvals

def get_padding_tight_z(ann, scan=None, scan_spacing=None, out_size_cm = None):
    """
    Get bbox dimensions base on a minimal size, restricting to no padding in z direction
    """
    if scan_spacing is None:
        if scan is None:
            scan_spacing = ann.scan.pixel_spacing
        else:
            scan_spacing = scan.pixel_spacing
    # return tight bounding box
    if out_size_cm is None:
        padding = [(int(0), int(0))] * 3
    else:
        # if len(out_size_cm == 3):
        #     if out_size_cm[2] is None:
        #         padding_z = (0,0)        
        out_shape = (np.ceil((out_size_cm * 10) / scan_spacing) * np.ones((2,))).astype(int)

        bb_mat   = ann.bbox_matrix()
        bb_shape = bb_mat[:,1] - bb_mat[:,0]

        paddings = out_shape - bb_shape[:2]

        # print(f"paddings: {paddings}")

        padding_x = (int(np.ceil(paddings[0] / 2)), int(np.floor(paddings[0] / 2)))
        padding_y = (int(np.ceil(paddings[1] / 2)), int(np.floor(paddings[1] / 2)))

        padding = [padding_x, padding_y, (int(0),int(0))]

    return padding


def resample_and_crop_annotation(ann_id, ann, nodule_path, mask_path=None, scan=None, size_mm = 50, export_mask = True):
    '''
    take an annotation, crop and resample
    size is the length of the sides of the resulting cube in millimeters
    '''
    if scan is None:
        scan = ann.scan
    intercept, slope = get_intercept_and_slope(scan)
    try:
        vol, mask = ann.uniform_cubic_resample(side_length = size_mm, verbose = True)
        if slope != 1:
            vol = slope * vol.astype(np.float64)

        vol = vol.astype(np.int16)
        vol += np.int16(intercept)
        
        np.save(os.path.join(nodule_path, ann_id+".npy"), vol)
        if export_mask:
            assert mask_path != None
            np.save(os.path.join(mask_path, ann_id+".npy"), mask)
        print("")
    except:
        print("-failed")

                
def flatten_multiindex_columns(df, sep = "_"):
    '''
    If a pandas DataFrame has a hierarchical index,
    flatten to single level
    '''
    col_vals = df.columns.values
    flattened = [sep.join(x) for x in col_vals]
    stripped = [x[:-1] if sep == x[-1] else x for x in flattened]
    df.columns = stripped
    return df
              
def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.
    
    Does not support timeout or chunksize as executor.submit is used internally
    
    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()

def tqdm_parallel_map_df(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.
    
    Does not support timeout or chunksize as executor.submit is used internally
    
    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for index, i in iterable]
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()

def normalize(x, window = None, level = None, in_min = -1000.0, in_max = 600.0, center=0.0):
    """
    Normalize array to values between 0 and 1, possibly clipping
    """
    assert type(x) is np.ndarray

    if (not window is None) & (not level is None) :
        in_min = level - (window / 2)
        in_max = level + (window / 2)

    x = x - in_min                 # add zero point
    x = x / (in_max - in_min)      # scale to unit
    x = x + center                 # adjust white-balance
    x = np.clip(x, 0.0, 1.0)       # clip to (0,1)
    return x

def normalized_to_8bit(x):
    assert ((x.min() >= 0) & (x.max() <= 1))
    x = (255 * x)
    return x.astype(np.uint8)

def normalize_to_8bit(x, *args, **kwargs):
    return normalized_to_8bit(normalize(x, *args, **kwargs))

def pwr_transform(x, train_ids=None):
    x  = np.array(x).reshape(-1,1)
    pt = PowerTransformer(method="yeo-johnson")
    if train_ids is None:
        pt.fit(x)
    else:
        pt.fit(x[train_ids])
    y = pt.transform(x)
    return np.squeeze(y)

def three_slice_rgb(x):
    '''
    Take 3 orthogonal slices from a cube and merge in single RGB image
    '''
    
    dim = x.shape
    assert (np.unique(dim)).shape == (1,)
    dim = dim[0]
    
    rgb = np.zeros((dim, dim, 3))
    
    rgb[:, :, 0] = x[np.int(dim / 2),:,:]
    rgb[:, :, 1] = x[:,np.int(dim / 2),:]
    rgb[:, :, 2] = x[:,:,np.int(dim / 2)]

    
    return rgb
    
def three_channel_plot(img, figsize=(12,6), titles=None):
    '''
    Take an RGB version of 3 orthogonal planes, and plot in 4 panels
    '''
    fig, axes = plt.subplots(1, 4, figsize = figsize)
    
    axes[0].imshow(normalize(img))
    axes[1].imshow(img[:, :, 0], cmap = "gray")
    axes[2].imshow(img[:, :, 1], cmap = "gray")
    axes[3].imshow(img[:, :, 2], cmap = "gray")
