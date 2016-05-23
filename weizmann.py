import sys, os
import cv2
import numpy as np
import imutils
from pylab import plt
from sklearn.externals.joblib import Parallel, delayed
from imutils.object_detection import non_max_suppression
from subprocess import call, PIPE
from skimage import io

# for i in bend jack pjump walk wave1 wave2; do mkdir $i; done
PATH_WEIZMANN = '/home/berthin/Dropbox/UNICAMP/Abnormal_Event_Detection/DataSets/weizmann/'
PATH_WEIZMANN_VR = '/home/berthin/Documents/weizmann-visual_rhythm-sobel/'
PATH_WEIZMANN_PATTERNS = \
    '/home/berthin/Documents/weizmann-visual_rhythm-patterns/'
WEIZMANN_CLASSES = ['bend', 'jack', 'pjump', 'walk', 'wave1', 'wave2']
WEIZMANN_CLASSES_INV = {WEIZMANN_CLASSES[idx]:idx for idx in xrange(len(WEIZMANN_CLASSES))}
WEIZMANN_NUMBER_CLASSES = 6
WEIZMANN_NUMBER_SAMPLES= 9

# for d in boxing handclapping handwaving walking; do mkdir $d; done
PATH_KTH = '/home/berthin/Documents/kth/'
PATH_KTH_VR = '/home/berthin/Documents/kth-visual_rhythm-sobel/'
PATH_KTH_PATTERNS = \
    '/home/berthin/Documents/kth-visual_rhythm-patterns/'
KTH_CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking']
KTH_CLASSES_INV = {KTH_CLASSES[idx]:idx for idx in xrange(len(KTH_CLASSES))}
KTH_NUMBER_CLASSES = 4

sys.path.append(os.environ['GIT_REPO'] + '/source-code/visual-rhythm')
import visual_rhythm
reload(visual_rhythm)

def read_weizmann_info (list_actions = 'all'):
    global PATH_WEIZMANN
    import re
    if list_actions == 'all':
        global WEIZMANN_CLASSES
        list_actions = WEIZMANN_CLASSES
    weizmann_info = {}
    for action in list_actions:
        for file_name in os.listdir(PATH_WEIZMANN + action):
            if file_name[-3:] == 'avi': weizmann_info[(action, file_name.split('_')[0]) ] = file_name[:-4]
    return weizmann_info


def read_kth_info (path_to_file, list_actions = 'all'):
    import re
    if list_actions == 'all':
        list_actions = ['boxing', 'handclapping', 'handwaving', 'walking']
    start_line = 21
    inFile = open(path_to_file)
    for i in xrange(start_line): inFile.readline()
    m_kth_info = {}
    while True:
        line = inFile.readline()
        if len(line) < 5 and ith_person == 25: break
        if len(line) < 5: continue
        action = re.findall('_[a-z]+_', line)[0][1:-1]
        if not action in list_actions: continue
        matches = np.array(map(int, re.findall('[0-9]+', line)))
        ith_person, ith_d = matches[:2]
        frame_intervals = matches[2:].reshape(len(matches)/2-1, 2)
        m_kth_info[(ith_person, action, ith_d)] = frame_intervals
    return m_kth_info

#according to an analysis, the min-max frame-interval is 68 among all the considered actions

#test using all the sequence
def get_visualrhythm_improved_whole_sequences(weizmann_info, type_visualrhythm = 'horizontal', params_vr = None, frame_size = (120, 160), frame_range=None,  n_frames = 10, sigma_canny = 2.5):
    from skimage import feature, morphology, filters
    global PATH_WEIZMANN
    global PATH_WEIZMANN_VR
    global PATH_WEIZMANN_PATTERNS
    start_frame, end_frame = frame_range

    for action, person in weizmann_info.items():
        action = action[0]
        cap = cv2.VideoCapture('%s%s/%s.avi' % (PATH_WEIZMANN, action, person))
        print PATH_WEIZMANN + action+ '/' + person
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print cap.isOpened()
        ith_frame = 0
        img_vr = []
        for counter in xrange(start_frame, end_frame + 1):
            if ith_frame > n_frames:break
            ith_frame +=1
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params_vr))
        if len(img_vr) < 10: break
        img_vr = np.array(img_vr)
        #img_vr = feature.canny(img_vr, sigma_canny) * 255
        #img_vr = cv2.medianBlur(img_vr, 3)
        print ('%s%s/%s.png' % (PATH_WEIZMANN_VR, action, person))
        cv2.imwrite('%s%s/%s.png' % (PATH_WEIZMANN_VR, action, person), img_vr)
        cv2.imwrite('%s%s/__%s.png' % (PATH_WEIZMANN_VR, action, person), cv2.normalize(filters.sobel(img_vr), None, 0, 255, cv2.NORM_MINMAX))


        thr_coe_var = 1.0
        ith_p = 1
        I = np.copy(img_vr)
        I = filters.sobel(I)
        #I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX)
        for ir in xrange(0, I.shape[1] - frame_size[1], frame_size[1]):
            patt = I[:, ir:ir+frame_size[1]]
            patt = patt[:, 20:frame_size[1]-20]
            coe_var = patt.std(axis=0) / patt.mean(axis=0)
            for col0 in xrange(0, patt.shape[1]):
                if coe_var[col0] > thr_coe_var: break
            for col1 in xrange(patt.shape[1]-1, 0, -1):
                if coe_var[col1] > thr_coe_var: break
            if (col1 - col0 + 1) < 10: continue
            cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), cv2.normalize(filters.sobel(patt[:, col0:col1+1]), None, 0, 255, cv2.NORM_MINMAX))
            ith_p += 1

        if False:
            for ir in xrange(0, img_vr.shape[1] - frame_size[1], frame_size[1]):
               img_vr[:, ir:ir+frame_size[1]] = feature.canny(img_vr[:, ir:ir+frame_size[1]], 1.5)
            cv2.imwrite('%s%s/_%s.png' % (PATH_WEIZMANN_VR, action, person), img_vr * 255)

            elem = np.zeros([7, 7]); elem[:, 7/2] = 1;
            img_vr =(img_vr - morphology.opening(img_vr, elem))
            img_vr = morphology.remove_small_objects(img_vr, min_size=10)

            cv2.imwrite('%s%s/+%s.png' % (PATH_WEIZMANN_VR, action, person), img_vr * 255)

            kernel_A = np.array([[0,0,1],[0,1,0],[1,0,0]], np.uint8)
            kernel_B = np.array([[1,0,0],[0,1,0],[0,0,1]], np.uint8)
            kernel_C = np.array([[0,0,0],[1,1,1],[0,0,0]], np.uint8)
            kernel_D = np.array([[0,1,0],[0,1,0],[0,1,0]], np.uint8)
            #kernels = [kernel_A, kernel_B]#, kernel_C, kernel_D]
            kernels = [kernel_A, kernel_B]

            for k in [kernel_A, kernel_B]:
                img_vr = img_vr - cv2.morphologyEx(img_vr, cv2.MORPH_DILATE, k)


            old = np.copy(img_vr)
            while True:
                for k in [kernel_A, kernel_B, kernel_C, kernel_D]:
                    img_vr = cv2.morphologyEx(img_vr, cv2.MORPH_CLOSE, k)
                if (old == img_vr).all(): break
                old = np.copy(img_vr)


            cv2.imwrite('%s%s/-%s.png' % (PATH_WEIZMANN_VR, action, person), img_vr * 255)

            cap.release()

def get_visualrhythm_sobel(weizmann_info, type_visualrhythm = 'horizontal', params_vr = None, frame_size = (120, 160), frame_range=None,  n_frames = 10):
    import skimage.feature
    import skimage.morphology
    import skimage.filters
    from skimage import measure
    from scipy import ndimage

    global PATH_WEIZMANN
    global PATH_WEIZMANN_VR
    global PATH_WEIZMANN_PATTERNS
    start_frame, end_frame = frame_range

    process_whole_video = end_frame == -1
    for action, person in weizmann_info.items():
        action = action[0]
        cap = cv2.VideoCapture('%s%s/%s.avi' % (PATH_WEIZMANN, action, person))
        print person
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ith_frame = 0
        img_vr = []
        if process_whole_video: end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        for counter in xrange(start_frame, end_frame + 1):
            if ith_frame > n_frames:break
            ith_frame +=1
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params_vr))
        if len(img_vr) < 10: break
        img_vr = np.array(img_vr)
        img_vr_sobel = skimage.filters.sobel(img_vr)
        img_vr_sobel_norm = cv2.normalize(img_vr_sobel, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imwrite('%s%s/%s.png' % (PATH_WEIZMANN_VR, action, person), img_vr_sobel_norm)

        ith_p = 1
        n_win = 9
        mask = np.ones([n_win, n_win]) / (1. * n_win * n_win)
        for ic in xrange(0, img_vr_sobel.shape[1] - frame_size[1], frame_size[1]):
            patt = img_vr_sobel[:, ic:ic+frame_size[1]]
            patt = patt[:, 20:frame_size[1]-20]
            _mean = ndimage.filters.convolve(patt, mask)
            _sd = (ndimage.filters.convolve(patt * patt, mask) - _mean * _mean)
            _coef = _mean / (1 + _sd)

            if _coef.max() < 0.1: continue
            cv2.imwrite('/tmp/patt.png', cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX))
            original = np.copy(patt)
            patt = _coef
            cv2.imwrite('/tmp/coef.png', cv2.normalize(_coef, None, 0, 255, cv2.NORM_MINMAX))
            print 'wrote'
            return True
            patt = cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, patt = cv2.threshold(patt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            for cmin in xrange(0, patt.shape[1]):
                if patt[:, cmin].max() > 0: break
            for cmax in xrange(patt.shape[1]-1, 0, -1):
                if patt[:, cmax].max() > 0: break
            n_con = (measure.label(patt)).max()
            if (cmax - cmin + 1) > 80 and n_con > 2: continue
            patt = patt[:, cmin: cmax+1]
            patt = cv2.normalize(original[:, cmin:cmax+1], None, 0, 255, cv2.NORM_MINMAX)
            patt = cv2.resize(patt, (100, 100), cv2.INTER_CUBIC)
            #patt = cv2.adaptiveThreshold(patt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
            cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), patt)
            #cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), _coef)
            ith_p += 1
            if ith_p == 4: break

def classify_sobel_weizmann(weizmann_info):
    from scipy import stats
    from sklearn import svm
    from sklearn import neighbors
    from sklearn import cross_validation
    from sklearn import metrics
    from skimage import feature

    global PATH_WEIZMANN_PATTERNS
    global WEIZMANN_CLASSES_INV
    global WEIZMANN_NUMBER_CLASSES
    global WEIZMANN_NUMBER_SAMPLES

    bag_features = [[] for i in xrange(WEIZMANN_NUMBER_CLASSES)]

    for action, person in weizmann_info.items():
        action = action[0]
        features_per_pattern = []
        for ith_p in xrange(1,4):
            pattern = cv2.imread('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), False)
            pattern_hog = feature.hog(pattern, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)).flatten()
            features_per_pattern.append(pattern_hog)
        bag_features[WEIZMANN_CLASSES_INV[action]].append(features_per_pattern)
    bag_features = np.array(bag_features)

    cv_indexes = cross_validation.LeaveOneOut(WEIZMANN_NUMBER_SAMPLES)
    hog_len = bag_features.shape[-1]
    acc_score = []
    answers = None
    labels = None
    for idx_train, idx_test in cv_indexes:
        train = bag_features[:, idx_train, ...]
        test = bag_features[:, idx_test, ...]

        label_train = np.repeat(range(1, WEIZMANN_NUMBER_CLASSES + 1), train.shape[1] * train.shape[2])
        label_test = range(1, WEIZMANN_NUMBER_CLASSES + 1)

        train = train.reshape(train.size / hog_len, hog_len)
        test = test.reshape(test.size / hog_len, hog_len)

        clf = svm.SVC(kernel='rbf', C=100).fit(train, label_train)
        #clf = svm.SVC(kernel='poly', degree=1).fit(train, label_train)
        #clf = neighbors.KNeighborsClassifier(n_neighbors = 1).fit(train, label_train)

        pred = clf.predict(test)
        pred = pred.reshape(pred.size/3, 3)

        output = []
        for ans in pred:
           output.append(stats.mode(ans)[0])

        acc_score.append(metrics.accuracy_score(label_test, output))
        output = np.array(output)
        label_test = np.array(label_test)
        answers = np.vstack((answers, output)) if answers is not None else output
        labels = np.hstack((labels, label_test)) if labels is not None else label_test
        #print metrics.accuracy_score(label_test, output)
    print acc_score
    answers = answers.flatten()
    labels = labels.flatten()
    print metrics.classification_report(answers, labels)
    print np.mean(acc_score)

def filter_patterns(x):
    #x = morphology.skeletonize((x > 0).astype(np.uint8))
    #x = morphology.closing(x, np.array([[0,1,0],[0,1,0],[0,1,0]]))
    x = 1 * (x > 0)
    x = np.vstack((x, np.zeros([1, x.shape[1]])))
    change = 0
    for col in xrange(10, x.shape[1], 10):
        last = 0
        for val in x[:, col]:
            change += 1 if last == 1 and val == 0 else 0
            last = val
    return change

def get_sum_vertical(x):
    x = np.uint8(x)
    return np.sum([filter_patterns(x & ((1 << i) | (1 << (i + 1)))) for i in xrange(5, 8)])

def get_visualrhythm_sobel_kth(kth_info, type_visualrhythm = 'horizontal', params_vr = None, frame_size = (120, 160), frame_range=None,  n_frames = 10, n_patterns = 5):
    import skimage.feature
    import skimage.morphology
    import skimage.filters
    from skimage import measure
    from scipy import ndimage

    global PATH_KTH
    global PATH_KTH_VR
    global PATH_KTH_PATTERNS

    for key, value in kth_info.items():
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        cap = cv2.VideoCapture('%s%s/person%02d_%s_d%d_uncomp.avi' % (PATH_KTH, action, ith_person, action, ith_d))
        ith_frame = 0
        ith_p = 1
        for start_frame, end_frame in value:
            img_vr = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for counter in xrange(start_frame, end_frame + 1):
                #if ith_frame > n_frames:break
                ith_frame +=1
                _, frame = cap.read()
                if frame is None: break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params_vr))
            if len(img_vr) < 10: break
            img_vr = np.array(img_vr)
            img_vr_sobel = skimage.filters.sobel(img_vr)
            img_vr_sobel_norm = cv2.normalize(img_vr_sobel, None, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite('%s%s/person%02d_%s_d%d_uncomp.png' % (PATH_KTH_VR, action, ith_person, action, ith_d), img_vr_sobel_norm)

            n_win = 9
            mask = np.ones([n_win, n_win]) / (1. * n_win * n_win)
            bag_patterns = []
            for ic in xrange(0, img_vr_sobel.shape[1] - frame_size[1], frame_size[1]):
                patt = img_vr_sobel[:, ic:ic+frame_size[1]]
                patt = patt[:, 20:frame_size[1]-20]
                _mean = ndimage.filters.convolve(patt, mask)
                _sd = (ndimage.filters.convolve(patt * patt, mask) - _mean * _mean)
                _coef = _mean / (1 + _sd)

                if _coef.max() < 0.1: continue
                original = np.copy(patt)
                patt = _coef
                #cv2.imwrite('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d, ith_p), cv2.normalize(_coef, None, 0, 255, cv2.NORM_MINMAX))
                patt = cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, patt = cv2.threshold(patt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                for cmin in xrange(0, patt.shape[1]):
                    if patt[:, cmin].max() > 0: break
                for cmax in xrange(patt.shape[1]-1, 0, -1):
                    if patt[:, cmax].max() > 0: break
                n_con = (measure.label(patt)).max()
                if (cmax - cmin + 1) > 80 and n_con > 2: continue
                patt = patt[:, cmin: cmax+1]
                patt = cv2.normalize(original[:, cmin:cmax+1], None, 0, 255, cv2.NORM_MINMAX)
                patt = cv2.resize(patt, (100, 100), cv2.INTER_CUBIC)
                #patt = cv2.adaptiveThreshold(patt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

                bag_patterns.append(patt);
                #cv2.imwrite('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d, ith_p), patt)
                #cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), _coef)
                ith_p += 1
                #if ith_p == 4: break
            bag_patterns.sort(key = get_sum_vertical, reverse = True)
            ith_p = 1
            for patt in bag_patterns[:n_patterns]:
                cv2.imwrite('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d, ith_p), patt)
                ith_p += 1



##########################
def extract_patterns_smart(action = 'boxing', nFrames = 50, frame_size = None, patt_size = (30, 50), n_patterns = 3, show = False, save_patterns = False, thr_std = 15):
    global PATH_KTH_VR
    global PATH_KTH_PATTERNS
    #
    init_time = time.time()
    #
    files = os.listdir(PATH_KTH_VR + action)
    files.sort()
    thr_std, thr_cvr = 15, 0.4
    thr_shape, thr_col = 10, 3
    #
    #filter_patterns = lambda x: x.flatten().sum()
    #only works for skeletons
    filter_patterns = lambda x: (np.sum([(x[:, col]>0).sum() for col in xrange(0, x.shape[1], 10)]))

    bag_patterns = []
    files.append('end_process12345')
    for im_name in files:
        im_0 = io.imread(PATH_KTH_VR + action + '/' + im_name, as_grey = True)
        #im_0 = im_0[:nFrames, :]
        patt_idx = 0
        if im_name[-5] != '1':
            bag_patterns = []
        elif len(bag_patterns) > 0:
            bag_patterns.sort(key = filter_patterns, reverse = True)
            ith_pattern = 0
            for pattern in bag_patterns[:n_patterns]:
                ith_pattern += 1
                io.imsave('%s%s/%s_p%d.bmp' % (PATH_KTH_PATTERNS, action, im_name[:-4], ith_pattern), pattern)
        if im_name[0] == 'e': break
        #find all patterns
        for im_1 in np.hsplit(im_0, im_0.shape[1] // frame_size[1]):
            last_col = -1
            pattern = np.empty([im_1.shape[0], 0], np.uint8)
            for col in xrange(im_1.shape[1]):
                if np.std (im_1[:, col]) < thr_std:
                    if (last_col != -1) and (col - last_col > thr_col):
                        pattern = np.hstack ((pattern, im_1[:, last_col:col]))
                    last_col = -1
                else:
                    last_col = col if last_col == -1 else last_col
            if last_col != -1 and col - last_col > thr_col:
                pattern = np.hstack ((pattern, im_1[:, last_col:col]))
            if pattern.shape[1] > thr_shape:
                if save_patterns:
                    bag_patterns.append(cv2.resize(pattern, patt_size, cv2.INTER_NEAREST))
                    #bag_patterns.append(pattern)
                patt_idx += 1

    #
    finish_time = time.time ()
    print (finish_time - init_time)


def extract_patterns_smart_var(action = 'boxing', nFrames = 50, frame_size = None, patt_size = (30, 50), param_VR = 5, n_patterns = 3, thr_min_pixels = 5, thr_min_gap = 10, show = False, save_patterns = False):
    global PATH_KTH_VR
    global PATH_KTH_PATTERNS
    #
    files = os.listdir(PATH_KTH_VR + action)
    files.sort()
    #
    #get_sum = lambda x: (x > 0).flatten().sum()
    #get_sum = lambda x: (x > 0).flatten().sum() * 1.0 /x.size
    #get_sum = lambda x: x.mean()
    def get_sum(x):
        mask = -1 * np.ones([3, 3], np.uint8)
        mask[1, :] = 2
        x = cv2.filter2D(x, cv2.CV_8U, mask)
        return x.mean()

    from skimage import morphology
    def filter_patterns(x):
        #x = morphology.skeletonize((x > 0).astype(np.uint8))
        #x = morphology.closing(x, np.array([[0,1,0],[0,1,0],[0,1,0]]))
        x = 1 * (x > 0)
        x = np.vstack((x, np.zeros([1, x.shape[1]])))
        change = 0
        for col in xrange(10, x.shape[1], 10):
            last = 0
            for val in x[:, col]:
                change += 1 if last == 1 and val == 0 else 0
                last = val
        return change
        #return np.sum([(x[:, col]).sum() for col in xrange(0, x.shape[1], 10)])

    bag_patterns = []
    files.append('end_process12345')
    last_im_name = ''
    for im_name in files:
        #im_0 = im_0[:nFrames, :]
        patt_idx = 0

        if im_name[-5] == '1' and len(bag_patterns) > 0:

            bag_patterns.sort(key = filter_patterns, reverse = True)
            ith_pattern = 0
            kernel_A = np.array([[0,0,1],[0,1,0],[1,0,0]], np.uint8)
            kernel_B = np.array([[1,0,0],[0,1,0],[0,0,1]], np.uint8)
            #kernel_C = np.array([[0,0,0],[1,1,1],[0,0,0]], np.uint8)
            #kernel_D = np.array([[0,1,0],[0,1,0],[0,1,0]], np.uint8)
            #kernels = [kernel_A, kernel_B]#, kernel_C, kernel_D]
            kernels = []

            print len(bag_patterns),
            for pattern in bag_patterns[:n_patterns]:
                ith_pattern += 1
                #pattern = cv2.resize(pattern, patt_size, cv2.INTER_CUBIC)
                pattern = np.array(pattern > 0, np.uint8)
                for k in kernels:
                    pattern = cv2.morphologyEx(pattern, cv2.MORPH_CLOSE, k)
                #pattern = morphology.skeletonize(pattern)

                io.imsave('%s%s/%s_p%d.bmp' % (PATH_KTH_PATTERNS, action, last_im_name[:-7], ith_pattern), 255*pattern)
            bag_patterns = []
        else:
            last_im_name = im_name

        if im_name[0] == 'e': break
        im_0 = io.imread(PATH_KTH_VR + action + '/' + im_name, as_grey = True)
        #extract patterns for same video
        for im_1 in np.hsplit(im_0, im_0.shape[1] // frame_size[1]):
            for col_1 in xrange(5, im_1.shape[1]):
                if im_1[:, col_1].sum() >= thr_min_pixels: break
            if col_1 + 1 >= im_1.shape[1]-5: continue
            for col_2 in xrange(im_1.shape[1]-5, col_1, -1):
                if im_1[:, col_2].sum() >= thr_min_pixels: break
            if col_2 == 0: continue
            if col_2 - col_1 + 1 < thr_min_gap: continue

            pattern = im_1[:, col_1:col_2+1]
            for row_1 in xrange(1, pattern.shape[0]):
                if pattern[row_1, :].sum() >= thr_min_pixels: break
            if row_1 + 1 >= im_1.shape[0]-5: continue
            for row_2 in xrange(pattern.shape[0]-5, row_1, -1):
                if pattern[row_2, :].sum() >= thr_min_pixels: break
            if row_2 == 0: continue
            if row_2 - row_1 + 1 < thr_min_gap: continue

            pattern = pattern[row_1:row_2+1, :]
            if pattern.size < 100*100*0.95: continue

            if save_patterns:
                bag_patterns.append(cv2.resize(pattern, patt_size, cv2.INTER_CUBIC))
                #bag_patterns.append(pattern)
        #ith_pattern = 0
        #for pattern in bag_patterns:
        #    ith_pattern += 1
        #    io.imsave('/tmp/test/%s_%s_p%d.bmp' % (action, im_name[:-4], ith_pattern), pattern)
        #break

    #

from skimage.feature import hog
def hog_per_action(action = 'boxing', actors_training = [], params_hog = None, n_patterns_per_video = 0):
    global PATH_KTH_PATTERNS
    data = []
    for ith_actor in actors_training:
        for ith_d in xrange(1, 5):
            for ith_p in xrange(1, 1 + n_patterns_per_video):
                pattern = cv2.imread('%s%s/person%02d_%s_d%d_p%d.bmp' % (PATH_KTH_PATTERNS, action, ith_actor, action, ith_d, ith_p), False)
                if pattern is None: continue
                pattern = (pattern > 0) * 255
                data.append(hog(pattern, **params_hog))
    return np.array(data)

#classify_visualrhythm_canny_patterns({'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}, 5)

def classify_visualrhythm_canny_patterns(params_hog, n_patterns_per_video):
    from skimage.feature import hog
    from sklearn.externals.joblib import Parallel, delayed

    global KTH_CLASSES

    #params_hog = {'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}
    #n_patterns_per_video = 5

    actors_training = [11, 12, 13, 14, 15, 16, 17, 18]
    actors_validation = [19, 20, 21, 23, 24, 25, 1, 4]
    actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    data_training_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_training, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_training = np.empty([0, data_training_list[0].shape[1]])
    label_training = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_training_list):
        data_training = np.vstack((data_training, datum))
        label_training = np.hstack((label_training, np.repeat(ith_action, datum.shape[0])))

    data_validation_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_validation, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_validation = np.empty([0, data_validation_list[0].shape[1]])
    label_validation = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_validation_list):
        data_validation = np.vstack((data_validation, datum))
        label_validation = np.hstack((label_validation, np.repeat(ith_action, datum.shape[0])))

    data_testing_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_testing, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_testing = np.empty([0, data_testing_list[0].shape[1]])
    label_testing = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_testing_list):
        data_testing = np.vstack((data_testing, datum))
        label_testing = np.hstack((label_testing, np.repeat(ith_action, datum.shape[0])))

    return (data_training, data_validation, data_testing,
            label_training, label_validation, label_testing)

import _hog
#use odd pixels_per_cell values
#tested with (5,5) and (7,7). fails when using (8,8) <-- TODO: check
def hog_variant_per_action(action = 'boxing', actors_training = [], params_hog = None, n_patterns_per_video = 0):
    global PATH_KTH_PATTERNS
    data = []
    for ith_actor in actors_training:
        for ith_d in xrange(1, 5):
            for ith_p in xrange(1, 1 + n_patterns_per_video):
                pattern = cv2.imread('%s%s/person%02d_%s_d%d_p%d.bmp' % (PATH_KTH_PATTERNS, action, ith_actor, action, ith_d, ith_p), False)
                if pattern is None: continue
                pattern = (pattern > 0) * 255
                data.append(_hog.hog_variant_superposition(pattern, **params_hog))
    return np.array(data)

#classify_visualrhythm_canny_patterns({'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}, 5)
def classify_visualrhythm_canny_patterns(params_hog, n_patterns_per_video):
    from skimage.feature import hog
    from sklearn.externals.joblib import Parallel, delayed

    global KTH_CLASSES

    #params_hog = {'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}
    #n_patterns_per_video = 5

    actors_training = [11, 12, 13, 14, 15, 16, 17, 18]
    actors_validation = [19, 20, 21, 23, 24, 25, 1, 4]
    actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    data_training_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_training, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_training = np.empty([0, data_training_list[0].shape[1]])
    label_training = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_training_list):
        data_training = np.vstack((data_training, datum))
        label_training = np.hstack((label_training, np.repeat(ith_action, datum.shape[0])))

    data_validation_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_validation, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_validation = np.empty([0, data_validation_list[0].shape[1]])
    label_validation = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_validation_list):
        data_validation = np.vstack((data_validation, datum))
        label_validation = np.hstack((label_validation, np.repeat(ith_action, datum.shape[0])))

    data_testing_list = Parallel(n_jobs=-1) (delayed(hog_per_action) (action, actors_testing, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_testing = np.empty([0, data_testing_list[0].shape[1]])
    label_testing = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_testing_list):
        data_testing = np.vstack((data_testing, datum))
        label_testing = np.hstack((label_testing, np.repeat(ith_action, datum.shape[0])))

    return (data_training, data_validation, data_testing,
            label_training, label_validation, label_testing)


sys.path.append(os.environ['GIT_REPO'] + '/source-code/bag-of-words')
import bag_of_words
def classify_bow_visualrhythm_canny_patterns(params_hog, params_bow, n_patterns_per_video, n_processors=-1, use_variant_bow = False):
    from sklearn.externals.joblib import Parallel, delayed

    global KTH_CLASSES

    #params_hog = {'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}
    #n_patterns_per_video = 5

    actors_training = [11, 12, 13, 14, 15, 16, 17, 18]
    actors_validation = [19, 20, 21, 23, 24, 25, 1, 4]
    actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    data_training_list = Parallel(n_jobs=n_processors) (delayed(hog_variant_per_action) (action, actors_training, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    #data_training_bow = None
    data_training_bow = np.empty([0, params_hog['orientations']])
    for ith_action, datum in zip(xrange(4), data_training_list):
        data_training_bow = np.vstack((data_training_bow, datum.reshape(datum.size/params_hog['orientations'], params_hog['orientations'])))

    cw, codebook_predictor = bag_of_words.build_codebook(data_training_bow, **params_bow)

    #training
    #data_training = np.empty([0, params_bow['number_of_words']])
    data_training = None
    label_training = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_training_list):
        if use_variant_bow:
            codes = []
            for pattern in datum:
                super_code = Parallel(n_jobs = n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], data_row.reshape(data_row.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for data_row in pattern)
                codes.append(np.array(super_code).flatten())
        else:
            codes = Parallel(n_jobs=n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], pattern.reshape(pattern.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for pattern in datum)
        print len(codes), codes[0].shape
        if data_training is None:
            data_training = codes
        else:
            data_training = np.vstack((data_training, codes))
        label_training = np.hstack((label_training, np.repeat(ith_action, datum.shape[0])))

    #validation
    data_validation_list = Parallel(n_jobs=n_processors) (delayed(hog_variant_per_action) (action, actors_validation, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_validation = None
    #data_validation = np.empty([0, params_bow['number_of_words']])
    label_validation = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_validation_list):
        if use_variant_bow:
            codes = []
            for pattern in datum:
                super_code = Parallel(n_jobs = n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], data_row.reshape(data_row.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for data_row in pattern)
                codes.append(np.array(super_code).flatten())
        else:
            codes = Parallel(n_jobs=n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], pattern.reshape(pattern.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for pattern in datum)
        if data_validation is None:
            data_validation = codes
        else:
            data_validation = np.vstack((data_validation, codes))
        label_validation = np.hstack((label_validation, np.repeat(ith_action, datum.shape[0])))

    #testing
    data_testing_list = Parallel(n_jobs=n_processors) (delayed(hog_variant_per_action) (action, actors_testing, params_hog, n_patterns_per_video) for action in KTH_CLASSES)
    data_testing = None
    #data_testing = np.empty([0, params_bow['number_of_words']])
    label_testing = np.empty([0])
    for ith_action, datum in zip(xrange(4), data_testing_list):
        if use_variant_bow:
            codes = []
            for pattern in datum:
                super_code = Parallel(n_jobs = n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], data_row.reshape(data_row.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for data_row in pattern)
                codes.append(np.array(super_code).flatten())
        else:
            codes = Parallel(n_jobs=n_processors)(delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor, params_bow['number_of_words'], pattern.reshape(pattern.size/params_hog['orientations'], params_hog['orientations']), params_bow['type_coding'], params_bow['type_pooling']) for pattern in datum)
        if data_testing is None:
            data_testing = codes
        else:
            data_testing = np.vstack((data_testing, codes))
        label_testing = np.hstack((label_testing, np.repeat(ith_action, datum.shape[0])))

    return (data_training, data_validation, data_testing,
            label_training, label_validation, label_testing)


def run_gridSearch(classifier, args, data_training, label_training, data_validation, label_validation):
    from sklearn import metrics
    return metrics.accuracy_score(classifier(**args).fit(data_training, label_training).predict(data_validation), label_validation)

def run_svm_canny_patterns (data_training, data_validation, data_testing,
        label_training, label_validation, label_testing, type_svm = 'svm'):
    from sklearn import svm, grid_search
    from sklearn import neighbors
    from sklearn import metrics

    _none = -1
    _svm = 0
    _nu_svm = 1
    _kernel_chi_square = 2
    _kernel_hik = 3
    _knn = 4
    _type_svm = _svm if type_svm == 'svm' else _nu_svm if type_svm == 'nu_svm' else _kernel_chi_square if type_svm == 'kernel_chi_square' else _kernel_hik if type_svm == 'kernel_hik' else _knn if type_svm == 'knn' else _none;

    ## Grid-search parameters setting

    gamma_range = np.power (10., np.arange (-5, 5, 0.5));
    C_range = np.power (10., np.arange (-5, 5));
    nu_range = np.arange (0.1, 0.6, 0.1);
    grid_search_params_SVC = \
      [{'kernel' : ['rbf'], 'C' : C_range, 'gamma' : gamma_range},\
       {'kernel' : ['linear'], 'C' : C_range}];
    grid_search_params_nuSVC = \
      [{'kernel' : ['rbf'], 'nu' : nu_range, 'gamma' : gamma_range},\
       {'kernel' : ['linear'], 'nu' : nu_range}];
    grid_search_params_kernel = \
      [{'kernel' : ['precomputed'], 'C' : C_range}];
    grid_search_params_knn = \
      [{'n_neighbors' : [1, 3, 5, 7, 9, 11]}]

    ## Temporaly we are gonna test only one-vs-one SVM linear-rbf kernels
    #elif _type_svm == _kernel_chi_square or _type_svm == _kernel_hik:
    #  grid_search = GridSearchCV (svm.SVC (), grid_search_params_kernel, cv=5, n_jobs=n_processors);

    if _type_svm == _svm:
        classifier = svm.SVC
        grid_search_params = grid_search_params_SVC
    elif _type_svm == _nu_svm:
        classifier = svm.NuSVC
        grid_search_params = grid_search_params_nuSVC
    elif _type_svm == _knn:
        classifier = neighbors.KNeighborsClassifier
        grid_search_params = grid_search_params_knn

    grid_search_ans = Parallel(n_jobs = -1)(delayed(run_gridSearch)(classifier, args, data_training, label_training, data_validation, label_validation) for args in list(grid_search.ParameterGrid(grid_search_params)))

#grid_search_ans = Parallel(n_jobs=-1)(delayed(metrics.accuracy_score)(label_validation, classifier(**args).fit(data_training, label_training).predict(data_validation) for args in list(grid_search.ParameterGrid(grid_search_params))))

    best_params = list(grid_search.ParameterGrid(grid_search_params))[grid_search_ans.index(max(grid_search_ans))]

    clf = classifier(**best_params).fit(np.vstack((data_training, data_validation)), np.hstack((label_training, label_validation)))

    pred = clf.predict(data_testing)

    print metrics.classification_report (pred, label_testing)

    print 'accuracy: ', metrics.accuracy_score(pred, label_testing)

    return pred


"""
n_classes = 5, n_samples_per_class = 10, n_features=-1, n_orientations = 9, n_words=300, type_coding = 'hard', type_pooling = 'max', type_svm = 'svm', n_processors=1):
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
"""


def get_visualrhythm_bounding_box_minmax(file_name, size, type_visualrhythm = 'horizontal', params = None, show = False, padding = 10, frame_size = (120, 160)):
    def extract_horizontal(img, (H, W), gap):
        return np.array([img[row, :] for row in xrange(0, H, gap)], np.uint8).flatten()
    def extract_vertical(img, (H, W), gap):
        return np.array([img[:, col] for col in xrange(0, W, gap)], np.uint8).flatten()
    def extract_zigzag(img, (H, W), row_gap, col_gap):
        return np.array(visual_rhythm.get_zigzag(frame, False, row_gap, col_gap, W, H)).flatten()

    global KTH_CLASSES_INV
    global PATH_KTH
    global PATH_KTH_OUT
    global PATH_KTH_VR
    #person number (1-25) + scenario number (1-4) + action number (1-6) (1-boxing, 2-hand clapping, 3-hand waving, 4-jogging, 5-running, 6-walking)+ sequence number (1-4) + start frame + end frame +             bounding box information (ymin xmin ymax xmax) for each frame
    inFile = open(file_name, 'r')
    inFile.readline()
    inFile.readline()
    counter = 0
    for i in xrange(0):
        inFile.readline()
        counter += 1
    seq = inFile.readline()
    while len(seq) > 0:
        arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
        video_name = 'person%02d_%s_d%d_uncomp.avi' % (arr[0], KTH_CLASSES[arr[2] - 1], arr[1])
        cap_in = cv2.VideoCapture(PATH_KTH + KTH_CLASSES[arr[2] - 1] + '/' + video_name)

        img_vr = []
        while True:
            print video_name
            idx_frame = 0
            counter += 1
            print counter
            arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
            start_frame, end_frame = arr[4:6]
            start_frame, end_frame = start_frame - 1, end_frame - 1
            cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            bounds = arr[6:].reshape(end_frame - start_frame + 1, 4)
            ymin, xmin, ymax, xmax = [np.min(bounds[:, i]) for i in xrange(4)]

            ymin = max(ymin - padding, 0)
            xmin = max(xmin - padding, 0)
            ymax = min(ymax + padding, frame_size[0] - 1)
            xmax = max(xmax + padding, frame_size[1] - 1)

            for ith_frame in xrange(end_frame-start_frame+1):
                ret, frame = cap_in.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame[ymin:ymax, xmin:xmax], size, cv2.INTER_CUBIC)
                if type_visualrhythm[0] == 'h': #horizontal
                    img_vr.append(extract_horizontal(frame, size, params[0]))
                elif type_visualrhythm[0] == 'v': #vertical
                    img_vr.append(extract_vertical(frame, size, params[0]))
                elif type_visualrhythm[0] == 'z': #zigzag
                    img_vr.append(extract_zigzag(frame, size, params[0], params[1]))
            seq = inFile.readline()
            prefix = ' '.join(map(str, arr[:3]))
            if not seq.startswith(prefix):
                break
        img_vr = np.array(img_vr)
        if show:
            plt.imshow(img_vr, cmap='gray')
            plt.show()

        cap_in.release()
        cv2.imwrite('%s%s/%s.png' % (PATH_KTH_VR, KTH_CLASSES[arr[2] - 1], video_name[:-4]), img_vr)

        break
    inFile.close()



#get_visualrhythm_bounding_box(PATH_KTH + 'bounding_box.in', (30, 50), 'horizontal', [10], False)
def get_visualrhythm_bounding_box(file_name, size, type_visualrhythm = 'horizontal', params = None, show = False):
    def extract_horizontal(img, (H, W), gap):
        return np.array([img[row, :] for row in xrange(0, H, gap)], np.uint8).flatten()
    def extract_vertical(img, (H, W), gap):
        return np.array([img[:, col] for col in xrange(0, W, gap)], np.uint8).flatten()
    def extract_zigzag(img, (H, W), row_gap, col_gap):
        return np.array(visual_rhythm.get_zigzag(frame, False, row_gap, col_gap, W, H)).flatten()

    global KTH_CLASSES_INV
    global PATH_KTH
    global PATH_KTH_OUT
    global PATH_KTH_VR
    #person number (1-25) + scenario number (1-4) + action number (1-6) (1-boxing, 2-hand clapping, 3-hand waving, 4-jogging, 5-running, 6-walking)+ sequence number (1-4) + start frame + end frame +             bounding box information (ymin xmin ymax xmax) for each frame
    inFile = open(file_name, 'r')
    inFile.readline()
    inFile.readline()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    counter = 0
    for i in xrange(0):
        inFile.readline()
        counter += 1
    seq = inFile.readline()
    while len(seq) > 0:
        arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
        video_name = 'person%02d_%s_d%d_uncomp.avi' % (arr[0], KTH_CLASSES[arr[2] - 1], arr[1])
        cap_in = cv2.VideoCapture(PATH_KTH + KTH_CLASSES[arr[2] - 1] + '/' + video_name)
        #cap_out = cv2.VideoWriter(PATH_KTH_OUT + KTH_CLASSES[arr[2] - 1] + '/' + video_name, fourcc, cap_in.get(cv2.CAP_PROP_FPS), size)

        img_vr = []
        while True:
            print video_name
            #call(['nohup', 'mkdir', '/tmp/' + video_name, '>', '/dev/null', '2>1'])
            idx_frame = 0
            counter += 1
            print counter
            arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
            start_frame, end_frame = arr[4:6]
            start_frame, end_frame = start_frame - 1, end_frame - 1
            cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for ymin, xmin, ymax, xmax in np.hsplit(arr[6:], end_frame-start_frame+1):
                ret, frame = cap_in.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame[ymin:ymax, xmin:xmax], size, cv2.INTER_CUBIC)
                if type_visualrhythm[0] == 'h': #horizontal
                    img_vr.append(extract_horizontal(frame, size, params[0]))
                elif type_visualrhythm[0] == 'v': #vertical
                    img_vr.append(extract_vertical(frame, size, params[0]))
                elif type_visualrhythm[0] == 'z': #zigzag
                    img_vr.append(extract_zigzag(frame, size, params[0], params[1]))
            seq = inFile.readline()
            prefix = ' '.join(map(str, arr[:3]))
            if not seq.startswith(prefix):
                break
        img_vr = np.array(img_vr)
        if show:
            plt.imshow(img_vr, cmap='gray')
            plt.show()

        cap_in.release()
        cv2.imwrite('%s%s/%s.png' % (PATH_KTH_VR, KTH_CLASSES[arr[2] - 1], video_name[:-4]), img_vr)

        #cap_out.release()
        break
    inFile.close()


#to run
#for action in ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']:
#    get_visualrhythm_whole_video(action, (120, 160), 'vertical', [5])
def get_visualrhythm_whole_video(action, size, type_visualrhythm = 'horizontal', params = None, show = False):
    def extract_horizontal(img, (H, W), gap):
        return np.array([img[row, :] for row in xrange(0, H, gap)], np.uint8).flatten()
    def extract_vertical(img, (H, W), gap):
        return np.array([img[:, col] for col in xrange(0, W, gap)], np.uint8).flatten()
    def extract_zigzag(img, (H, W), row_gap, col_gap):
        return np.array(visual_rhythm.get_zigzag(frame, False, row_gap, col_gap, W, H)).flatten()
    #
    global PATH_KTH
    global PATH_KTH_VR
    #
    list_files = [file for file in os.listdir(PATH_KTH + '/' + action) if file[0] is not '.' and file[-3:] == 'avi']
    for video_name in list_files:
        cap = cv2.VideoCapture(PATH_KTH + '/' + action + '/' + video_name)
        img_vr = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if type_visualrhythm[0] == 'h': #horizontal
                img_vr.append(extract_horizontal(frame, size, params[0]))
            elif type_visualrhythm[0] == 'v': #vertical
                img_vr.append(extract_vertical(frame, size, params[0]))
            elif type_visualrhythm[0] == 'z': #zigzag
                img_vr.append(extract_zigzag(frame, size, params[0], params[1]))
        img_vr = np.array(img_vr)
        if show:
            plt.imshow(img_vr, cmap = 'gray')
            plt.show()
        cap.release()
        cv2.imwrite('%s%s/%s.png' % (PATH_KTH_VR, action, video_name[:-4]), img_vr)

