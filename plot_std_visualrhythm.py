import pandas
import numpy as np
import pylab as plt
from skimage import io

source = weizmann.PATH_WEIZMANN_VR
img = io.imread (source + 'wave1/__denis_wave1.png', as_grey=True)
#im1 = img[:, 180*1:180*15] / 255.
im1 = img / 255.

mean = np.mean (im1, axis=0)
std = np.std (im1, axis=0)

mean = None
std = None

im2 = np.empty([im1.shape[0], 0])
gap1, gap2 = 40, 120
for i in xrange(0, im1.shape[1] - 180, 180):
    if mean is None:
        mean = im1[:, i+gap1:i+gap2].mean(axis=0)
        std = im1[:, i+gap1:i+gap2].std(axis=0)
        im2 = im1[:, i+gap1:i+gap2]
    else:
        mean = np.hstack((mean, im1[:, i+gap1:i+gap2].mean(axis=0)))
        std = np.hstack((std, im1[:, i+gap1:i+gap2].std(axis=0)))
        im2 = np.hstack((im2, im1[:, i+gap1:i+gap2]))

df = pandas.DataFrame (np.vstack ((std, std/mean)).T, columns=['Std', 'Coeff. of Variation'])

df.plot (kind='area', stacked=False, figsize=(27, 5), legend=True)

io.imsave('/tmp/myfig.png', im2)
plt.savefig ('/tmp/myfig.pdf', bbox_inches='tight')

