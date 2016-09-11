from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy.misc import imsave
from scipy import ndimage as ndi
import os

src_path1 = 'data/luk'
dist_path1 = 'data/cropped_luk'


def calc_bounds(img):
    img = rgb2gray(img)
    edges = canny(img, sigma=0.5)

    fill_potatoes = ndi.binary_fill_holes(edges)
    label_objects, nb_labels = ndi.label(fill_potatoes)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    potatoes_cleaned = mask_sizes[label_objects]

    ax0 = potatoes_cleaned.sum(axis=0)[150:-150]
    ax1 = potatoes_cleaned.sum(axis=1)[150:-150]
    ax00 = ax0.argmax(axis=None)
    ax10 = ax1.argmax(axis=None)

    s = np.linspace(0, 2 * np.pi, 400)
    x = ax00 + 150 * np.cos(s)
    y = ax10 + 150 * np.sin(s)
    # init = np.array([x, y]).T

    lex = ax00 - 150 + 150
    rex = ax00 + 150 + 150
    ley = ax10 - 150 + 150
    hey = ax10 + 150 + 150
    return lex, rex, ley, hey

def edge_dir(src_path, dist_path):
    imgs_paths = os.listdir(src_path)
    for img_path in imgs_paths:
        img = io.imread(os.path.join(src_path, img_path), as_grey=False)
        lex, rex, ley, hey = calc_bounds(img)
        cropped_img = img[ley:hey, lex:rex, :]
        imsave(os.path.join(dist_path, img_path), cropped_img)


'''
plt.plot(range(len(ax0)), ax0)
plt.plot(range(len(ax1)), ax1)
print (ax00, ax10)'''
'''
snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, w_edge=5, gamma=0.001)
fig, (ax, ax1, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
#fig = plt.figure(figsize=(7, 7))
#ax = fig.add_subplot(111)
#ax1 = fig.add_subplot(111)
plt.gray()
ax1.imshow(potatoes_cleaned)
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
plt.show()
'''