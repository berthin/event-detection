import numpy as np
import cv2

import kth_opticalFlow

PATH_KTH_PATTERNS = \
    '/home/berthin/Documents/kth-visual_rhythm-horizontal-canny-patterns/'

n_orientations = int(raw_input('orientations: '))
pixels_per_cell = map(int, raw_input('pixels_per_cell: ').split(' '))
cells_per_block = map(int, raw_input('cells_per_block: ').split(' '))
n_patterns_per_video = int(raw_input('n_patterns_per_video: '))

params_hog = {'orientations':n_orientations,
              'pixels_per_cell':(pixels_per_cell[0], pixels_per_cell[1]),
              'cells_per_block':(cells_per_block[0],cells_per_block[1]),
              'visualise':False}

type_svm = raw_input('type_svm: ')
use_bow = raw_input('use bow: ')

if use_bow[0] == 'Y' or use_bow[0]=='1':
    n_words = int(raw_input('number_of_words: '))
    clustering_method = raw_input('clustering_method: ')
    type_coding = raw_input('type_coding: ')
    type_pooling = raw_input('type_pooling: ')
    use_variant_bow = bool(raw_input('use variant bow: ')) #simple do not enter anything

    params_bow = {'number_of_words':n_words, 'clustering_method':clustering_method, 'type_coding':type_coding, 'type_pooling':type_pooling}

if False:
    params_bow = {'number_of_words':300, 'clustering_method':'random', 'type_coding':'hard', 'type_pooling':'avg'}

    params_hog = {'orientations':9,
              'pixels_per_cell':(7,7),
              'cells_per_block':(2,2),
              'visualise':False}

if use_bow[0] == 'Y' or use_bow[0]=='1':
    (data_training, data_validation, data_testing,
        label_training, label_validation, label_testing) = \
        kth_opticalFlow.classify_bow_visualrhythm_canny_patterns(params_hog, params_bow,
        n_patterns_per_video, -1, use_variant_bow)
else:
    (data_training, data_validation, data_testing,
        label_training, label_validation, label_testing) = \
        kth_opticalFlow.classify_visualrhythm_canny_patterns(params_hog,
        n_patterns_per_video, -1, use_variant_bow)

ans1 = kth_opticalFlow.run_svm_canny_patterns(data_training, data_validation, data_testing, label_training, label_validation, label_testing, type_svm)


from scipy import stats
ans2 = np.array([stats.mode(x)[0] for x in np.array(ans1).reshape(9*4*4, n_patterns_per_video)]).flatten()
label = np.hstack([np.ones(9*4)*i for i in xrange(4)]).flatten()

from sklearn import metrics
print metrics.classification_report(ans2, label)
print metrics.accuracy_score(ans2, label)
