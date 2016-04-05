import os, sys, time
import cv2
import numpy as np
from skimage import io, feature
from scipy import stats
from matplotlib import pylab as plt
from sklearn.decomposition import sparse_encode
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import Normalizer as norm
from sklearn.decomposition import *

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from _hog import *

sys.path.append(os.environ['GIT_REPO'] + '/source-code/bag-of-words')
import bag_of_words
reload (sys.modules.get ('bag_of_words'))
reload (sys.modules.get ('_hog'))

"""
> data: n_classes x n_actors x n_samples_per_actor x n_patterns_per_sample x 15 x 8 x n_orientations
"""
#Parser for bag-of-words
def parse_data1 (data, n_orientations, indexes, n_classes, n_scenarios):
    d2 = np.empty ([0, n_orientations], np.float)
    l2 = np.empty ([0], np.uint8)
    for ith_class in xrange(n_classes):
        for actor in indexes:
            for ith_scenario in xrange(n_scenarios):
                for pattern in data[ith_class][actor][ith_scenario]:
                    d2 = np.vstack((d2, pattern.reshape(pattern.size / n_orientations, n_orientations)))
                    l2 = np.hstack ((l2, np.repeat (ith_class, pattern.size / n_orientations)))
    return (d2, l2)

from new_classifier import unname_classifier
reload (sys.modules.get ('new_classifier'))
#Parser for bag-of-words
def parse_data6 (data, n_orientations, indexes, n_classes, n_scenarios, codebook_predictor, n_words, type_coding, type_pooling):
    d2, l2 = [], []
    for ith_class in xrange(n_classes):
        for actor in indexes:
            for ith_scenario in xrange(n_scenarios):
                #f1 = Parallel (n_jobs = -1) (delayed (bag_of_words.coding_pooling_per_video) (codebook_predictor, n_words, data[ith_class][actor][ith_scenario][idx_pattern].reshape(data[ith_class][actor][ith_scenario][idx_pattern].size / n_orientations, n_orientations), type_coding, type_pooling) for idx_pattern in xrange(len(data[ith_class][actor][ith_scenario])))
                #for f11 in f1:
                #    d2.append(f11.ravel())
                #    l2.append(ith_class)
                for pattern in data[ith_class][actor][ith_scenario]:
                    f1 = bag_of_words.coding_pooling_per_video (codebook_predictor, n_words, pattern.reshape (pattern.size / n_orientations, n_orientations), type_coding, type_pooling)
                    f1 = np.array (f1)
                    d2.append(f1.ravel())
                    l2.append(ith_class)
    d2 = np.array(d2)
    l2 = np.array(l2)
    return (d2, l2)

##
training_time = 0
def run_newclassifier_globaldescriptor (data, n_classes = 5, n_samples_per_class = 10, n_features=-1, n_orientations = 9, n_words=300, type_coding = 'hard', type_pooling = 'max', type_svm = 'svm', n_processors=1):
  debug = True
  global training_time
  #for debugging
  n_classes, n_scenarios, n_words = 4, 4, 300
  type_coding, type_pooling = 'hard', 'mean'

  pred_total = np.empty (0, np.uint8);
  label_total = np.empty (0, np.uint8);

  actors_training = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 1, 4]
  actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]
  train = np.array(actors_training, np.uint8) - 1
  test = np.array(actors_testing, np.uint8) - 1
  if True:
    ## BAG OF WORDS
    pre_data_train, _ = parse_data1 (data, n_orientations, train, n_classes, n_scenarios)

    #pca = PCA (n_components = 3).fit_transform (pre_data_train)
    #plt.scatter (pca[:, 0], pca[:, 1])
    #fig = plt.figure()
    #ax = fig.add_subplot (111, projection = '3d')
    #ax.scatter (xs=pca[:, 0], ys=pca[:, 1], zs=pca[:, 2])
    #plt.show ()

    time_a = time.time ()
    cw, codebook_predictor = bag_of_words.build_codebook (pre_data_train, number_of_words = n_words, clustering_method = 'kmeans')

    #data_train, label_train = Parallel (n_jobs = n_processors) (delayed (parse_data6) (data, n_orientations, idx_train, codebook_predictor, n_words, type_coding, type_pooling, n_samples_per_class) for idx_train in train)
    #data_test, label_test = Parallel (n_jobs = n_processors) (delayed (parse_data6) (data, n_orientations, idx_test, codebook_predictor, n_words, type_coding, type_pooling, n_samples_per_class) for idx_test in test)
    data_train, label_train = parse_data6 (data, n_orientations, train, n_classes, n_scenarios, codebook_predictor, n_words, type_coding, type_pooling)

    data_test, label_test = parse_data6 (data, n_orientations, test, n_classes, n_scenarios, codebook_predictor, n_words, type_coding, type_pooling)
    time_b = time.time ()
    training_time += (time_b - time_a)
    print 'data codes obtained'
    #data_train = np.array (data_train)
    #data_test = np.array (data_test)

    #data_tmp = []
    #for d1 in data_train:
    #  for d2 in d1:
    #    data_tmp.append (d2)
    #data_tmp = np.array (data_tmp)
    #pca = PCA()
    #pca.fit (data_tmp)
    #print pca.explained_variance_ratio_.cumsum()
    #pca.n_components = np.searchsorted (pca.explained_variance_ratio_.cumsum() > 0.97, True)

    #data_train, label_train = parse_data5 (data, n_features, train, n_samples_per_class)
    #data_test, label_test = parse_data5 (data, n_features, test, n_samples_per_class)

    #if debug:
    #  return (data_train, data_test)

    #print data_train.shape, data_test.shape

    ## Temporaly we are gonna test only one-vs-one SVM linear-rbf kernels
    clf = unname_classifier (data_train, label_train, distance = 'tanimoto')
    clf.fit (n_processors = 8)

    #grid_search.fit (data_train, label_train);
    #best_params = grid_search.best_params_;
    #best_score = grid_search.best_score_;
    #print best_params

    #clf = grid_search.estimator;
    #clf.set_params (**best_params);
    #clf.fit (data_train, label_train);

    pred = clf.predict (data_test, C=1000000., n_processors = 8);

    pred_total = np.hstack ((pred_total, pred));
    label_total = np.hstack ((label_total, label_test));

    #print np.array_str (pred, max_line_width=200)
    #print np.array_str (label_test, max_line_width=200)

    #print classification_report (label_test, pred)
  print label_total
  print pred_total
  print classification_report (label_total, pred_total)
  return (None, None)


if __name__ == '__main__':
  global training_time
  PATH_KTH_PATTERNS = '/home/berthin/Documents/kth-patterns-horizontal/'
  training_time = 0
  init_time = time.time ()

  n_classes = 4
  actors_training = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 1, 4]
  actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]
  n_samples = 4
  n_actors = 25
  n_orientations = 6
  map_label = {'boxing':0, 'handclapping':1, 'handwaving':2, 'running':3}
  #data = np.empty ([n_classes, n_samples, 0, 15, 8, n_orientations], np.float )
  data = [[[[] for i in xrange(n_samples)] for j in xrange(n_actors)] for k in xrange(n_classes)]
  #data = [[[] for i in xrange (n_samples)] for y in xrange (n_classes)]
  data_boxes = np.empty ([0, n_orientations], np.float);
  # Read data
  for action in map_label.keys():
    list_files = os.listdir(PATH_KTH_PATTERNS + action)
    for file_name, idx in zip (list_files, range (len (list_files))):
      if not 'bmp' in file_name: continue
      im1 = io.imread (PATH_KTH_PATTERNS + action + '/' + file_name, as_grey = True)
      features_im1 = hog_variant_superposition (im1, orientations=n_orientations, pixels_per_cell=(5,5), cells_per_block=(2,2))
      class_im1 = map_label[action]
      ith_actor = int(file_name[6:8]) - 1
      sample_im1 = ord (file_name[file_name.find ('_d') + 2]) - ord('0') - 1
      data[class_im1][ith_actor][sample_im1].append (features_im1)

  #Run SVM
  data = np.array (data)
  n_features = len(data[0][0][0][0])
  data_train, data_test = run_newclassifier_globaldescriptor (data, n_classes, n_samples, n_features, n_orientations, n_words=25, type_coding = 'hard', type_pooling = 'sum', type_svm = 'svm', n_processors=8)
  finish_time = time.time ()
  total_time = (finish_time - init_time)

  print 'total time: ', total_time
  print 'training_time: ', (training_time), 'time:', (total_time - training_time)


