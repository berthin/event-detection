import numpy as np
import cv2

import kth_opticalFlow

PATH_KTH_PATTERNS = \
    '/home/berthin/Documents/kth-visual_rhythm-horizontal-improved-patterns/'

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

kth_opticalFlow.run_svm_canny_patterns(data_training, data_validation,
    data_testing, label_training, label_validation, label_testing, type_svm)

