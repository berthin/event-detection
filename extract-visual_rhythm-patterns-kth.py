from skimage import io
import numpy as np
import os
import time

def extract_patterns(action = 'boxing', size = None, show = False, save_patterns = False):
    global PATH_KTH_VR
    global PATH_KTH_PATTERNS
    #
    init_time = time.time()
    #
    files = os.listdir(PATH_KTH_VR + action)
    thr_std, thr_cvr = 10, 0.4
    thr_shape, thr_col = 10, 10

    for im_name in files:
        im_0 = io.imread(PATH_KTH_VR + action + '/' + im_name, as_grey = True)
        patt_idx = 0
        for im_1 in np.hsplit(im_0, im_0.shape[1] // size[1]):
            #patt_idx += 1
            last_col = -1
            pattern = np.empty([im_1.shape[0], 0], np.uint8)
            for col in xrange(im_1.shape[1]):
                if np.std (im_1[:, col]) < thr_std:
                #if np.std (im_1[:, col]) < thr_cvr * np.std (im_1[:, col]):
                    if (last_col != -1) and (col - last_col > thr_std):
                        pattern = np.hstack ((pattern, im_1[:, last_col:col]))
                        #io.imsave ('%s%s_img=%d_p%d.bmp' % (dest, im_name[:-4], im_idx, patt_idx), im_1[:, last_col:col])
                        #patt_idx += 1
                    last_col = -1
                else:
                    last_col = col if last_col == -1 else last_col
            if last_col != -1 and col - last_col > thr_col:
                pattern = np.hstack ((pattern, im_1[:, last_col:col]))
                #io.imsave ('%s%s_img=%d_p%d.bmp' % (dest, im_name[:-4], im_idx, patt_idx), im_1[:, last_col:col])
            if pattern.shape[1] > thr_shape:
                if show:
                    plt.imshow(pattern, cmap='gray')
                    plt.show()
                if save_patterns:
                    io.imsave('%s%s/%s_p%d.bmp' % (PATH_KTH_PATTERNS, action, im_name[:-4], patt_idx), pattern)
                patt_idx += 1
    #
    finish_time = time.time ()
    print (finish_time - init_time)
