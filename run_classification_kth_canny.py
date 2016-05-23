import numpy as np
import cv2

import kth_opticalFlow
import weizmann

#PATH_KTH_PATTERNS = \
#    '/home/berthin/Documents/kth-patterns_t1/kth-visual_rhythm-canny-patterns-morphology/'

kth_opticalFlow.PATH_KTH_PATTERNS = weizmann.PATH_KTH_PATTERNS

n_orientations = int(raw_input('orientations: '))
pixels_per_cell = map(int, raw_input('pixels_per_cell: ').split(' '))
cells_per_block = map(int, raw_input('cells_per_block: ').split(' '))
n_patterns_per_video = int(raw_input('n_patterns_per_video: '))
type_svm = raw_input('type_svm: ')

params_hog = {'orientations':n_orientations,
              'pixels_per_cell':(pixels_per_cell[0], pixels_per_cell[1]),
              'cells_per_block':(cells_per_block[0],cells_per_block[1]),
              'visualise':False}

(data_training, data_validation, data_testing,
    label_training, label_validation, label_testing) = \
    kth_opticalFlow.classify_visualrhythm_canny_patterns(params_hog,
    n_patterns_per_video)

ans1 = kth_opticalFlow.run_svm_canny_patterns(data_training, data_validation,
      data_testing, label_training, label_validation, label_testing, type_svm)


from scipy import stats
ans2 = np.array([stats.mode(x)[0] for x in np.array(ans1).reshape(9*4*4, n_patterns_per_video)]).flatten()
label = np.hstack([np.ones(9*4)*i for i in xrange(4)]).flatten()

from sklearn import metrics
print metrics.classification_report(ans2, label)
print metrics.accuracy_score(ans2, label)
