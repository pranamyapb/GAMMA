import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.linear_model import LinearRegression

def get_background_mask(img, sigma_clip=3.0):
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    sigma = 1.4826 * mad

    mask = img < (med + sigma_clip * sigma)

    mask = ndimage.binary_erosion(mask, iterations=1)

    if np.sum(mask) < 50:  
        print("⚠️ Background mask empty — relaxing threshold...")
        mask = img < (med + 5 * sigma)
        mask = ndimage.binary_erosion(mask, iterations=1)

    # If still empty → use *upper* half of histogram
    if np.sum(mask) < 50:
        print("⚠️ Still empty — using top 70% darkest pixels instead.")
        thresh = np.percentile(img, 70)
        mask = img < thresh

    return mask


def histogram_distance(img1, img2, mask):
    vals1 = img1[mask].ravel()
    vals2 = img2[mask].ravel()
    return wasserstein_distance(vals1, vals2)

def compare_backgrounds(path1, path2):
    # Read images as float
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

    mask = get_background_mask(img1)

    emd = histogram_distance(img1, img2, mask)


    vals1 = img1[mask].ravel()
    vals2 = img2[mask].ravel()

    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.imshow(img1, cmap='gray'); plt.title('GAMMA'); plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(img2, cmap='gray'); plt.title('Montage Method (Approx.)'); plt.axis('off')

    plt.subplot(2,1,2)
    plt.hist(vals1, bins=100, alpha=0.5, label='GAMMA', density=True)
    plt.hist(vals2, bins=100, alpha=0.5, label='Montage', density=True)
    plt.legend(); plt.title('Background Pixel Intensity Histogram')
    plt.xlabel('Pixel Value'); plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig('background_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


compare_backgrounds('my_img.png', 'output_image.png')
