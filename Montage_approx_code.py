import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import griddata

def remove_small_oabjects(mask, min_size):
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_clean = np.zeros_like(mask)
    for i, size in enumerate(sizes):
        if size >= min_size:
            mask_clean[label_im == i] = 1
    return mask_clean.astype(bool)

def estimate_background(image, sigma_clip=3.0, obj_size_thresh=50, interp_method='linear'):
    """
    Estimate and subtract (or return) background for an astronomical image using
    small‐object removal + interpolation.
    
    Params:
      image: 2D numpy array (float32) of the image (single channel).
      sigma_clip: threshold in sigma for marking objects above background.
      obj_size_thresh: minimum pixel‐area to keep as background region (objects removed).
      interp_method: interpolation for missing pixels ('linear', 'cubic', 'nearest').
      
    Returns:
      background: 2D array of estimated background same size as image.
    """
    h, w = image.shape
    
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51))
    smooth = cv2.morphologyEx(image, cv2.MORPH_OPEN, selem)
    
    residual = image - smooth
    
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    sigma = 1.4826 * mad
    obj_mask = residual > (sigma_clip * sigma)
    
    background_mask = ~obj_mask
    background_mask = remove_small_objects(background_mask, obj_size_thresh)
    
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.vstack((xx[background_mask].ravel(), yy[background_mask].ravel())).T
    vals = image[background_mask].ravel()
    
    grid_z0 = griddata(pts, vals, (xx, yy), method=interp_method, fill_value=np.nan)
    
    nanmask = np.isnan(grid_z0)
    if np.any(nanmask):
        grid_z0[nanmask] = griddata(pts, vals, (xx[nanmask], yy[nanmask]), method='nearest')
    
    background = grid_z0.astype(np.float32)
    return background

img = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
bkg = estimate_background(img, sigma_clip=3.0, obj_size_thresh=100, interp_method='cubic')
adjusted = img - bkg + np.median(bkg)
#adjusted[adjusted<70] = 0
adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
cv2.imwrite("output_image.png", adjusted)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1,2,2); plt.imshow(adjusted, cmap='gray'); plt.title('Background-corrected')

plt.show()
