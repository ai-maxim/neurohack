from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.feature import canny

img = io.imread('/home/xenakrll/Pictures/kartoxa/potato/1473498619.jpg', as_grey=True)
img = io.imread('/home/xenakrll/Pictures/kartoxa/potato/1473498361.jpg', as_grey=True)
img = io.imread('/home/xenakrll/Pictures/kartoxa/potato/1473498375.jpg', as_grey=True)
img = io.imread('/home/xenakrll/Pictures/kartoxa/data/train/potato/WP_20160911_115.jpg', as_grey=True)
img = io.imread('/home/xenakrll/Pictures/kartoxa/data/train/potato/WP_20160911_043.jpg', as_grey=True)



edges = canny(img, sigma=0.75)
from scipy import ndimage as ndi
fill_potatoes = ndi.binary_fill_holes(edges)
label_objects, nb_labels = ndi.label(fill_potatoes)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 15
mask_sizes[0] = 0
potatoes_cleaned = mask_sizes[label_objects]

ax0 = potatoes_cleaned.sum(axis=0)[50:-50]
ax1 = potatoes_cleaned.sum(axis=1)[50:-50]
ax00 = ax0.argmax(axis=None)
ax10 = ax1.argmax(axis=None)

#plt.plot(range(len(ax0)), ax0)
#plt.plot(range(len(ax1)), ax1)
print (ax00, ax10)

s = np.linspace(0, 2*np.pi, 400)
x = ax00 + 200*np.cos(s)
y = ax10 + 150*np.sin(s)
init = np.array([x, y]).T


snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, w_edge=5, gamma=0.001)
fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
#fig = plt.figure(figsize=(7, 7))
#ax = fig.add_subplot(111)
#ax1 = fig.add_subplot(111)
plt.gray()
#ax1.imshow(potatoes_cleaned)
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
plt.show()