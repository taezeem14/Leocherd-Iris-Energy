import cv2
from pathlib import Path
import numpy as np

img_path = Path("data/samples/creature_reference.jpg")
img = cv2.imread(str(img_path))
h, w, c = img.shape
print(f"Creature Image: {w}x{h}")

# Background is white/light gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Invert so creature is foreground
_, thresh = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY_INV)

# Find largest contour (the creature)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(largest)
print(f"Creature Bounding Box: x={x}, y={y}, w={bw}, h={bh} ({bw/bh:.2f} aspect ratio)")

# Horizontal profile analysis: width at every Y slice
widths = []
centers_x = []
for row in range(y, y + bh):
    slice_fg = np.where(thresh[row, x:x+bw] > 0)[0]
    if len(slice_fg) > 0:
        w_row = slice_fg[-1] - slice_fg[0]
        c_x = x + (slice_fg[-1] + slice_fg[0]) * 0.5
        widths.append(w_row)
        centers_x.append(c_x)
    else:
        widths.append(0)
        centers_x.append(x + bw * 0.5)

widths = np.array(widths)
# Normalize Y from 0.0 (head top) to 1.0 (feet bottom)
head_slice = widths[:int(0.25*bh)]
shoulder_slice = widths[int(0.20*bh):int(0.35*bh)]
torso_slice = widths[int(0.35*bh):int(0.55*bh)]
hips_slice = widths[int(0.55*bh):int(0.65*bh)]
legs_slice = widths[int(0.65*bh):]

print(f"Max Shoulder Width: {np.max(shoulder_slice):.1f}px ({np.max(shoulder_slice)/bh:.2f} of height)")
print(f"Mid Torso Width   : {np.mean(torso_slice):.1f}px ({np.mean(torso_slice)/bh:.2f} of height)")
print(f"Hips Width        : {np.mean(hips_slice):.1f}px")
print(f"Legs / Boots Width: {np.max(legs_slice):.1f}px")

# Stocky/Dwarf Morphology Detection:
# Standard human shoulder/height ratio is ~0.25-0.28.
# This dwarf is > 0.65 (extremely broad & stocky)!
ratio = np.max(shoulder_slice) / float(bh)
print(f"Detected Morphology Classification: {'Brute / Dwarf Heavy Biped' if ratio > 0.40 else 'Standard Human'}")
