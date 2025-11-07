import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: Load image
img = cv2.imread('im2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
h, w = gray.shape

# Step 2: Detect vertical and horizontal peaks
grad_cols = np.array([np.sum(gray[:, i+1] - gray[:, i]) for i in range(w-1)])
prom_cols = 0.5 * np.max(np.abs(grad_cols))
pos_peaks_cols, _ = find_peaks(grad_cols, prominence=prom_cols)
neg_peaks_cols, _ = find_peaks(-grad_cols, prominence=prom_cols)
all_peaks_cols = np.sort(np.concatenate([pos_peaks_cols, neg_peaks_cols]))
all_peaks_cols = np.concatenate(([0], all_peaks_cols, [w-1]))
print(grad_cols)
grad_rows = np.array([np.sum(gray[i+1, :] - gray[i, :]) for i in range(h-1)])
prom_rows = 0.5 * np.max(np.abs(grad_rows))
pos_peaks_rows, _ = find_peaks(grad_rows, prominence=prom_rows)
neg_peaks_rows, _ = find_peaks(-grad_rows, prominence=prom_rows)
all_peaks_rows = np.sort(np.concatenate([pos_peaks_rows, neg_peaks_rows]))
all_peaks_rows = np.concatenate(([0], all_peaks_rows, [h-1]))
# ---- Visualization of Column Gradient Peaks ----
plt.figure(figsize=(12,4))
plt.plot(grad_cols, label="Column Intensity Gradient")
plt.scatter(pos_peaks_cols, grad_cols[pos_peaks_cols], color='red', label='Positive Peaks')
plt.scatter(neg_peaks_cols, grad_cols[neg_peaks_cols], color='blue', label='Negative Peaks')
plt.title("Vertical (Column-wise) Intensity Change & Peaks")
plt.xlabel("Column Index")
plt.ylabel("Gradient Value")
plt.legend()
plt.grid(True)
plt.show()

# ---- Visualization of Row Gradient Peaks ----
plt.figure(figsize=(12,4))
plt.plot(grad_rows, label="Row Intensity Gradient")
plt.scatter(pos_peaks_rows, grad_rows[pos_peaks_rows], color='red', label='Positive Peaks')
plt.scatter(neg_peaks_rows, grad_rows[neg_peaks_rows], color='blue', label='Negative Peaks')
plt.title("Horizontal (Row-wise) Intensity Change & Peaks")
plt.xlabel("Row Index")
plt.ylabel("Gradient Value")
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Target mean for background
# 
target_mean = np.median(gray)  # Use median to avoid stars influencing the mean

# Step 4: Adjust intensity per box (preserve stars)
gray_adjusted = gray.copy()
bright_thresh = np.percentile(gray, 99)  # Pixels brighter than 99th percentile considered stars

for r in range(len(all_peaks_rows)-1):
    for c in range(len(all_peaks_cols)-1):
        row_start, row_end = all_peaks_rows[r], all_peaks_rows[r+1]
        col_start, col_end = all_peaks_cols[c], all_peaks_cols[c+1]

        box = gray[row_start:row_end, col_start:col_end]
        # Mask bright stars
        background_mask = box < bright_thresh
        if np.sum(background_mask) == 0:
            continue  # Skip box if everything is bright

        box_bg = box[background_mask]
        box_bg_mean = np.mean(box_bg)
        shift = target_mean - box_bg_mean

        # Apply shift only to background pixels
        box_shifted = box.copy()
        box_shifted[background_mask] += shift
        box_shifted = np.clip(box_shifted, 0, 255)

        gray_adjusted[row_start:row_end, col_start:col_end] = box_shifted

# Step 5: Optional: highlight box boundaries
img_highlight = cv2.cvtColor(gray_adjusted.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for col in all_peaks_cols:
    cv2.line(img_highlight, (col,0), (col,h-1), (0,0,255), 1)
for row in all_peaks_rows:
    cv2.line(img_highlight, (0,row), (w-1,row), (0,255,0), 1)
#gray_adjusted[gray_adjusted<70] = 0
# Step 6: Display results
plt.figure(figsize=(8, 12))  # Taller figure to accommodate two images vertically
gray_adjusted = gray_adjusted.astype(np.uint8)
cv2.imwrite('my_img.png',gray_adjusted)
plt.subplot(3, 1, 1) 
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(3, 1, 2)  
plt.imshow(img_highlight, cmap='gray')
plt.title("BOX highlights")
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(gray_adjusted, cmap='gray')
plt.title("Adjusted Background (Stars Preserved)")
plt.axis('off')
plt.tight_layout()
plt.savefig("comparison_original_vs_adjusted_vertical2.png", dpi=300, bbox_inches='tight')
plt.show()
