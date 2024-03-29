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
PATH_KTH_OPTICALFLOW = '/home/berthin/Documents/kth-optical_flow/'
PATH_KTH_OPTICALFLOW_PATTERNS = '/home/berthin/Documents/kth-optical_flow-patterns/'
PATH_KTH_VR = '/home/berthin/Documents/kth-visual_rhythm-sobel/'
PATH_KTH_PATTERNS = \
    '/home/berthin/Documents/kth-visual_rhythm-patterns-new/'
KTH_CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']
KTH_CLASSES_INV = {KTH_CLASSES[idx]:idx for idx in xrange(len(KTH_CLASSES))}
KTH_NUMBER_CLASSES = 6

PATH_SHEFFIELD = '/home/berthin/Documents/SheffieldKinectGesture/'
PATH_SHEFFIELD_VR = '/home/berthin/Documents/sheffield-visual_rhythm-sobel/'
PATH_SHEFFIELD_PATTERNS = \
    '/home/berthin/Documents/sheffield-visual_rhythm-patterns/'
SHEFFIELD_CLASSES = [None, 'circle', 'triangle', 'updown', 'rightleft', 'wave', 'z', 'cross', 'comehere', 'turnaround', 'pat']
SHEFFIELD_CLASSES_INV = {SHEFFIELD_CLASSES[idx]:idx for idx in xrange(len(SHEFFIELD_CLASSES))}
SHEFFIELD_NUMBER_CLASSES = 10
SHEFFIELD_VIDEO_MODE = {'K':'dep', 'M':'rgb'}


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

def search_in_kth_info (m_kth_info, (ith_person, action, ith_d)):
    import re
    param_ith_person = '[0-9]+' if not ith_person else str(ith_person)
    param_action = '[a-z]+' if not action else action
    param_ith_d = '[0-9]+' if not ith_d else str(ith_d)
    return [(key, value) for key, value in m_kth_info.items() if re.search('_%s_%s_%s_' % (param_ith_person, param_action, param_ith_d), '_%s_' %'_'.join(map(str, key)))]


def search_in_sheffield_info (sheffield_info, (video_mode, ith_person, background, illumination, pose, action)):

    #video_mode, ith_person, background, illumination, pose, action = video_details
    import re
    param_video_mode = '[M,K]' if not video_mode else video_mode
    param_ith_person = '[0-9]+' if not ith_person else str(ith_person)
    param_background = '[0-9]' if not background else background
    param_illumination = '[0-9]' if not illumination else illumination
    param_pose = '[0-9]' if not pose else pose
    param_action = '[0-9]+' if not action else action
    return [(key, value) for key, value in sheffield_info.items() if re.search('%s_person_%s_backgroud_%s_illumination_%s_pose_%s_actionType_%s.avi' % (param_video_mode, param_ith_person, param_background, param_illumination, param_pose, param_action), value)]

def read_kth_info (path_to_file, list_actions = 'all'):
    import re
    if list_actions == 'all':
        global KTH_CLASSES
        list_actions = KTH_CLASSES
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


def read_sheffield_info (list_actions = 'all', source_mode = ['rgb', 'dep']):
    global PATH_SHEFFIELD
    import re

    global SHEFFIELD_CLASSES
    global SHEFFIELD_CLASSES_INV

    if list_actions == 'all':
        list_actions = map(SHEFFIELD_CLASSES_INV.get, SHEFFIELD_CLASSES[1:])
    else:
        list_actions = map(SHEFFIELD_CLASSES_INV.get, list_actions)

    sheffield_info = {}
    def parserInt(x):
        try:
            return int(x)
        except Exception:
            return str(x)

    for mode in source_mode:
        for ith_person in xrange(1, 7):
            for file_name in os.listdir(PATH_SHEFFIELD + 'subject%d_%s' % (ith_person, mode)):
                file_name = file_name[:-4]
                param_names = re.findall('[a-z]+[a-zA-Z]+', file_name)
                param_values = map(parserInt, re.split('[a-z, _]+[a-zA-Z, _]+', file_name))
                sheffield_info[tuple(param_values)] = file_name + '.avi'
    return sheffield_info
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

def get_sum_vertical(patt):
    x = patt.getPattern()
    if type(x).__name__ == 'tuple':
        x = x[0]
    x = np.uint8(x)
    return np.sum([filter_patterns(x & ((1 << i) | (1 << (i + 1)))) for i in xrange(5, 8)])

class Pattern:
    # Init function params must follow: posR = (initR, lenR), posC = (initC, lenC), posT = (initT, lenT)
    def __init__(self, img, posR, posC, posT):
        self.img = img
        self.posR = posR
        self.posC = posC
        self.posT = posT

    def getPattern(self):
        return self.img
    def getPosR(self):
        return self.posR
    def getPosC(self):
        return self.posC
    def getPosT(self):
        return self.posT

def get_visualrhythm_sobel_kth(kth_info, params_filter_kth, type_visualrhythm = 'horizontal', params_vr = None, frame_size = (120, 160), n_patterns = 5):
    import skimage.feature
    import skimage.morphology
    import skimage.filters
    from skimage import measure
    from scipy import ndimage

    global PATH_KTH
    global PATH_KTH_VR
    global PATH_KTH_PATTERNS

    thr_coef = 0.07
    #for key, value in kth_info.items():
    for key, value in search_in_kth_info(kth_info, params_filter_kth):
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        cap = cv2.VideoCapture('%s%s/person%02d_%s_d%d_uncomp.avi' % (PATH_KTH, action, ith_person, action, ith_d))
        ith_frame = 0
        ith_p = 1
        bag_patterns = []
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

            #cv2.imwrite('%s%s/person%02d_%s_d%d_uncomp.png' % (PATH_KTH_VR, action, ith_person, action, ith_d), img_vr_sobel_norm)

            n_win = 9
            mask = np.ones([n_win, n_win]) / (1. * n_win * n_win)
            #works only for horizontal & vertical, by the moment only for horizontal
            initR, lenR = -params_vr[0], params_vr[0]
            initC, lenC = 0, frame_size[1]
            initT, lenT = start_frame, end_frame - start_frame + 1
            for ic in xrange(0, img_vr_sobel.shape[1] - frame_size[1], frame_size[1]):
                initR += params_vr[0]
                patt = img_vr_sobel[:, ic:ic+frame_size[1]]
                patt = patt[:, 20:frame_size[1]-20]
                _mean = ndimage.filters.convolve(patt, mask)
                _sd = (ndimage.filters.convolve(patt * patt, mask) - _mean * _mean)
                _coef = _mean / (1 + _sd)

                if _coef.max() < thr_coef: continue
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
                initC, lenC = cmin + 20, cmax - cmin + 1 #fixing bug
                #patt = patt[:, cmin: cmax+1]
                patt = cv2.normalize(original[:, cmin:cmax+1], None, 0, 255, cv2.NORM_MINMAX)
                patt = cv2.resize(patt, (100, 100), cv2.INTER_CUBIC)
                #patt = cv2.adaptiveThreshold(patt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

                bag_patterns.append(Pattern(patt, (initR, lenR), (initC, lenC), (initT, lenT)));
                #cv2.imwrite('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d, ith_p), patt)
                #cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), _coef)
        if len(bag_patterns) < 5:
            print action, ith_person, ith_d, len(bag_patterns)
        bag_patterns.sort(key = get_sum_vertical, reverse = True)
        ith_p = 1
        for patt in bag_patterns[:n_patterns]:
            cv2.imwrite('%s%s/person%02d_%s_d%d_p%d_(%d_%d)(%d_%d)(%d_%d).png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d, ith_p, patt.getPosR()[0], patt.getPosR()[1], patt.getPosC()[0], patt.getPosC()[1], patt.getPosT()[0], patt.getPosT()[1]), patt.getPattern())
            ith_p += 1

def check_patterns_sheffield(sheffield_info, params_filter_sheffield, n_patterns = 5):
    import os
    global PATH_SHEFFIELD_PATTERNS

    for video_details, file_name in search_in_sheffield_info(sheffield_info, params_filter_sheffield):
        video_mode, ith_person, background, illumination, pose, action = video_details
        for ith_p in xrange(1, n_patterns + 1):
            if not os.path.isfile('%s%s/%s_p%d.png' % (PATH_SHEFFIELD_PATTERNS, action, file_name[:-4], ith_p)):
                print file_name[:-4]
                break


#weizmann.get_visualrhythm_sobel_sheffield(sheffield_info, ('M', 1,1,1,1,1), type_visualrhythm = 'horizontal', params_vr = [5], frame_size = (240, 320), frame_range=(-1, -1),  n_frames = 10, n_patterns = 5, debug = False):
def get_visualrhythm_sobel_sheffield(sheffield_info, params_filter_sheffield, type_visualrhythm = 'horizontal', params_vr = None, frame_size = (240, 320), frame_range=None,  n_frames = 10, n_patterns = 5, debug = False):
    import skimage.feature
    import skimage.morphology
    import skimage.filters
    from skimage import measure
    from scipy import ndimage

    global PATH_SHEFFIELD
    global PATH_SHEFFIELD_VR
    global PATH_SHEFFIELD_PATTERNS
    global SHEFFIELD_VIDEO_MODE

    start_frame, end_frame = frame_range
    process_whole_video = end_frame == -1
    for video_details, file_name in search_in_sheffield_info(sheffield_info, params_filter_sheffield):
        video_mode, ith_person, background, illumination, pose, action = video_details

        cap = cv2.VideoCapture('%s/subject%d_%s/%s' % (PATH_SHEFFIELD, ith_person, SHEFFIELD_VIDEO_MODE[video_mode], file_name))
        capK = cv2.VideoCapture('%s/subject%d_%s/%s' % (PATH_SHEFFIELD, ith_person, SHEFFIELD_VIDEO_MODE['K'], file_name.replace('M', 'K')))
        print file_name
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ith_frame = 0
        img_vr = []
        img_vr_bw = []
        if process_whole_video: end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        for counter in xrange(start_frame, end_frame + 1):
            if ith_frame > n_frames:break
            ith_frame +=1
            _, frame = cap.read()
            _, frameK = capK.read()
            if not _: break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frameK = 255 * (cv2.cvtColor(frameK, cv2.COLOR_RGB2GRAY) < 240)
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[..., 2]
            frame = np.bitwise_and(frameK, frame)
            img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params_vr))
            img_vr_bw.append(visual_rhythm.extract_from_frame(frameK, type_visualrhythm, frame_size, params_vr))
        if len(img_vr) < 10: break
        img_vr = np.array(img_vr)
        img_vr_bw = np.array(img_vr_bw)

        #cv2.imwrite('%s%s/_%s.png' % (PATH_SHEFFIELD_VR, action, file_name[:-4]), img_vr)
        img_vr_sobel = skimage.filters.sobel(img_vr)
        img_vr_sobel_norm = cv2.normalize(img_vr_sobel, None, 0, 255, cv2.NORM_MINMAX)

        if debug:
            cv2.imwrite('/tmp/%s_vr.png' % file_name[:-4], img_vr)
            cv2.imwrite('/tmp/%s_vr_bw.png' % file_name[:-4], img_vr_bw)
            cv2.imwrite('/tmp/%s_sobel.png' % file_name[:-4], img_vr_sobel_norm)

        ith_p = 1
        n_win = 9
        thr_coef = 0.01
        thr_conn = 1.5
        mask = np.ones([n_win, n_win]) / (1. * n_win * n_win)
        bag_patterns = []
        for ic in xrange(0, img_vr_sobel.shape[1] - frame_size[1], frame_size[1]):
            patt = img_vr_sobel[:, ic:ic+frame_size[1]]
            patt = patt[:, 20:frame_size[1]-20]
            patt_bw = img_vr_bw[:, ic+20:ic+frame_size[1]-20]

            patt_bw = cv2.medianBlur(patt_bw, 3)
            for cmin in xrange(0, patt_bw.shape[1]):
                if patt_bw[:, cmin].max() > 0: break
            for cmax in xrange(patt_bw.shape[1]-1, 0, -1):
                if patt_bw[:, cmax].max() > 0: break
            if (cmax - cmin + 1) < 10: continue
            patt_bw = patt_bw[:, cmin: cmax+1]
            patt = patt[:, cmin:cmax+1]

            """
            _mean = ndimage.filters.convolve(patt, mask)
            _sd = (ndimage.filters.convolve(patt * patt, mask) - _mean * _mean)
            _coef = _mean / (1 + _sd)

            if debug:
                img_debug = np.vstack((
                    cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX),
                    cv2.normalize(_coef, None, 0, 255, cv2.NORM_MINMAX),
                    cv2.threshold(cv2.normalize(_coef, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                ))

            if _coef.max() < thr_coef: continue
            #cv2.imwrite('/tmp/patt_she%d.png'%ith_p, cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX))
            original = np.copy(patt)
            patt = _coef

            #print 'wrote'
            #return True
            patt = cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, patt = cv2.threshold(patt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            for cmin in xrange(0, patt.shape[1]):
                if patt[:, cmin].max() > 0: break
            for cmax in xrange(patt.shape[1]-1, 0, -1):
                if patt[:, cmax].max() > 0: break
            if (cmax - cmin + 1) < 10: continue
            #patt = patt[:, cmin: cmax+1]
            conn = measure.label(patt)
            n_con = conn.max()
            ans = [skimage.morphology.convex_hull_image(conn == i).sum()/1.0/(conn == i).sum() for i in range(n_con+1)]
            idx_valid_conn = [i for i,j in enumerate(ans) if j > thr_conn and patt[conn == i].max() > 0]
            valid_conn = np.zeros(patt.shape, np.bool)
            for idx in idx_valid_conn:
                valid_conn = np.bitwise_or(valid_conn, conn == idx)
            patt = np.bitwise_and(patt, valid_conn)

            if debug:
                img_debug = np.vstack((img_debug, patt * 255))
                cv2.imwrite('/tmp/%s_p%d_coef=%f.png' % (file_name[:-4], ith_p, _coef.max()), img_debug)
                #cv2.imwrite('/tmp/%s_p%d_ncon_%d.png' % (file_name[:-4], ith_p, n_con), patt)

            #if (cmax - cmin + 1) > 80 and n_con > 2: continue
            patt = cv2.normalize(original[:, cmin:cmax+1], None, 0, 255, cv2.NORM_MINMAX)

            """
            patt = cv2.normalize(patt, None, 0, 255, cv2.NORM_MINMAX)
            patt = cv2.resize(patt, (100, 100), cv2.INTER_CUBIC)

            patt_bw = cv2.normalize(patt_bw, None, 0, 255, cv2.NORM_MINMAX)
            patt_bw = cv2.resize(patt_bw, (100, 100), cv2.INTER_CUBIC)
            if debug:
                cv2.imwrite('/tmp/%s_p%d.png' % (file_name[:-4], ith_p), patt)
            #patt = cv2.adaptiveThreshold(patt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
            #last#cv2.imwrite('%s%s/%s_p%d.png' % (PATH_SHEFFIELD_PATTERNS, action, file_name[:-4], ith_p), patt)
            #if debug:
            #    cv2.imwrite('/tmp/%s_p%d.png' % (file_name[:-4], ith_p), patt)

            bag_patterns.append((patt_bw, patt))
            #cv2.imwrite('%s%s/%s_p%d.png' % (PATH_WEIZMANN_PATTERNS, action, person, ith_p), _coef)
            ith_p += 1
            #if ith_p == 4: break

        bag_patterns.sort(key = get_sum_vertical, reverse = True)
        ith_p = 6
        for patt_bw, patt in bag_patterns[5:n_patterns]:
            cv2.imwrite('%s%s/%s_p%d.png' % (PATH_SHEFFIELD_PATTERNS, action, file_name[:-4], ith_p), patt)
            ith_p += 1
        #return True

def extract_nHog_features(params_hog, *params_pattern):
    from skimage import feature
    pattern = cv2.imread('%s%d/M_person_%d_backgroud_%d_illumination_%d_pose_%d_actionType_%d_p%d.png' % (params_pattern), False)
    pattern_hog = feature.hog(pattern, **params_hog).flatten()
    return pattern_hog

def get_features_sheffield_per_person(params_hog, n_patterns_per_video, ith_action):
    from skimage import feature
    global PATH_SHEFFIELD_PATTERNS
    data = []
    for ith_background in xrange(1, 4):
        for ith_illumination in xrange(1, 3):
            for ith_pose in xrange(1, 4):
                for ith_person in xrange(1, 7):
                    patterns_hog = Parallel(n_jobs=1)(delayed(extract_nHog_features)(params_hog, PATH_SHEFFIELD_PATTERNS, ith_action, ith_person, ith_background, ith_illumination, ith_pose, ith_action, ith_p) for ith_p in xrange(1, n_patterns_per_video + 1))
                    data.append(np.array(patterns_hog))

    return data


def get_features_sheffield(params_hog, n_patterns_per_video):
    data = Parallel(n_jobs=-1)(delayed(get_features_sheffield_per_person)(params_hog, n_patterns_per_video, ith_action) for ith_action in xrange(1, 11))
    data = np.array(data)
    return data
    #return data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3])

def classify_sheffield_simple_fold(idx_train, idx_test, data, idx, params_svm, n_patterns):
    train = [data[ith_class, idx[idx_train], ...] for ith_class in xrange(10)]
    train = np.array(train)
    train = train.reshape(train.size / data.shape[-1], data.shape[-1])
    label_train = np.repeat(range(1, 11), n_patterns * 2 * len(idx) / 3)

    test = [data[ith_class, idx[idx_test], ...] for ith_class in xrange(10)]
    test = np.array(test)
    test = test.reshape(test.size / data.shape[-1], data.shape[-1])
    label_test = np.repeat(range(1, 11), len(idx) / 3)

    clf = svm.SVC(**params_svm).fit(train, label_train)
    #clf = svm.SVC(kernel='poly', degree=1).fit(train, label_train)
    #clf = neighbors.KNeighborsClassifier(n_neighbors = 1).fit(train, label_train)

    pred = clf.predict(test)
    pred = pred.reshape(pred.size/n_patterns, n_patterns)

    output = []
    for ans in pred:
       output.append(stats.mode(ans)[0])

    accuracy = metrics.accuracy_score(label_test, output)
    output = np.array(output)
    label_test = np.array(label_test)
    return (accuracy, output, label_test)


from sklearn import cross_validation
from sklearn import svm
from scipy import stats
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import metrics
from skimage import feature


#classify_sheffield(params_hog = {'orientations':9, 'pixels_per_cell':(8,8),'cells_per_block':(2,2)}, params_svm = {'kernel':'rbf', 'C':100}, n_patterns = 5)
def classify_sheffield(params_hog, params_svm, n_patterns):
    data = get_features_sheffield(params_hog, n_patterns)


    idx = np.arange(data.shape[1])
    np.random.shuffle(idx)
    kf = cross_validation.KFold(data.shape[1], n_folds=3)
    labels = None
    answers = None
    acc_score = []
    tmp_ans = Parallel(n_jobs=-1)(delayed(classify_sheffield_simple_fold)(idx_train, idx_test, data, idx, params_svm, n_patterns) for idx_train, idx_test in kf)

    acc_score = [tmp_ans[i][0] for i in range(3)]
    answers = np.array([tmp_ans[i][1].flatten() for i in range(3)]).flatten()
    labels = np.array([tmp_ans[i][2].flatten() for i in range(3)]).flatten()
    print acc_score
    print metrics.classification_report(answers, labels)
    print np.mean(acc_score)

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
    import glob
    global PATH_KTH_PATTERNS
    global KTH_CLASSES_INV

    data = []
    label = []
    for ith_actor in actors_training:
        for ith_d in xrange(1, 5):
            for ith_p in xrange(1, 1 + n_patterns_per_video):
                file_path = glob.glob('%s%s/person%02d_%s_d%d_p%d*.png' % (PATH_KTH_PATTERNS, action, ith_actor, action, ith_d, ith_p))
                if not file_path: continue
                pattern = cv2.imread(file_path[0], False)
                #pattern = (pattern > 0) * 255
                data.append(hog(pattern, **params_hog))
                label.append(KTH_CLASSES_INV[action])
    return (np.array(data), np.array(label))

def hof_per_action(action = 'boxing', actors_training = [], params_hof = None, n_patterns_per_video = 0):
    import _hog
    reload(_hog)
    import glob
    global PATH_KTH_OPTICALFLOW_PATTERNS
    global KTH_CLASSES_INV

    data = []
    label = []
    for ith_actor in actors_training:
        for ith_d in xrange(1, 5):
            for ith_p in xrange(1, 1 + n_patterns_per_video):
                file_path = glob.glob('%s%s/person%02d_%s_d%d_p%d*.npz' % (PATH_KTH_OPTICALFLOW_PATTERNS, action, ith_actor, action, ith_d, ith_p))
                if not file_path: continue
                mag, ang = np.load(file_path[0])['arr_0']
                data.append(_hog.hof2(mag, ang, **params_hof))
                label.append(KTH_CLASSES_INV[action])
    return (np.array(data), np.array(label))
#classify_visualrhythm_canny_patterns({'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}, 5)

def get_features_kth(descriptor, params_hogf, n_patterns_per_video, list_actions = None):
    from sklearn.externals.joblib import Parallel, delayed

    if descriptor == 'hog':
        feature_extractor = hog_per_action
    elif descriptor == 'hof':
        feature_extractor = hof_per_action

    global KTH_CLASSES
    global KTH_NUMBER_CLASSES
    if list_actions == None:
        list_actions = KTH_CLASSES
        n_actions = KTH_NUMBER_CLASSES
    else:
        n_actions = len(list_actions)

    #params_hog = {'orientations':8, 'pixels_per_cell':(8, 8), 'cells_per_block':(2,2), 'visualise':False}
    #n_patterns_per_video = 5

    actors_training = [11, 12, 13, 14, 15, 16, 17, 18]
    actors_validation = [19, 20, 21, 23, 24, 25, 1, 4]
    actors_testing = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    datainfo = (Parallel(n_jobs=-1) (delayed(feature_extractor) (action, actors_training, params_hogf, n_patterns_per_video) for action in list_actions))
    data_training_list = np.array([datainfo[i][0] for i in range(n_actions)])
    if len(data_training_list.shape) == 1:
        data_training_list, label_training_list = np.hsplit(np.array(datainfo), 2)
        data_training_list = data_training_list.reshape(-1)
        data_training = np.empty([0, data_training_list[0].shape[1]])
        label_training_list = label_training_list.reshape(-1)
        label_training = np.empty([0])
        for ith_action, datum in zip(xrange(n_actions), data_training_list):
            data_training = np.vstack((data_training, datum))
            label_training = np.hstack((label_training, label_training_list[ith_action]))
    else:
        label_training = np.array([datainfo[i][1] for i in range(n_actions)]).flatten()
        data_training = data_training_list.reshape(-1, data_training_list.shape[-1])
        data_training_list = data_training_list.reshape(-1)

    datainfo = (Parallel(n_jobs=-1) (delayed(feature_extractor) (action, actors_validation, params_hogf, n_patterns_per_video) for action in list_actions))
    data_validation_list = np.array([datainfo[i][0] for i in range(n_actions)])
    label_validation = np.array([datainfo[i][1] for i in range(n_actions)]).flatten()
    data_validation = data_validation_list.reshape(-1, data_validation_list.shape[-1])

    datainfo = (Parallel(n_jobs=-1) (delayed(feature_extractor) (action, actors_testing, params_hogf, n_patterns_per_video) for action in list_actions))
    data_testing_list = np.array([datainfo[i][0] for i in range(n_actions)])
    label_testing = np.array([datainfo[i][1] for i in range(n_actions)]).flatten()
    data_testing = data_testing_list.reshape(-1, data_testing_list.shape[-1])
    #data_testing_list = data_testing_list.reshape(-1)
    #data_testing = np.empty([0, data_testing_list[0].shape[1]])
    ##label_testing = np.empty([0])
    #label_testing = label_testing.flatten()
    #for ith_action, datum in zip(xrange(n_actions), data_testing_list):
    #    data_testing = np.vstack((data_testing, datum))
    #    #label_testing = np.hstack((label_testing, np.repeat(ith_action, datum.shape[0])))

    return (data_training, data_validation, data_testing,
            label_training, label_validation, label_testing)

"""
weizmann.classify_sobel_kth(descriptor='hof', params_hogf=dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)), params_svm=dict(kernel='rbf', C=100), n_patterns=5, type_svm='svm', list_actions=weizmann.KTH_CLASSES)
"""
def classify_sobel_kth(descriptor, params_hogf, params_svm, n_patterns=5, type_svm='svm', list_actions = None, return_pred = False):

    if descriptor == 'hog-hof':
        data_training_hog, data_validation_hog, data_testing_hog, label_training_hog, label_validation_hog, label_testing_hog = get_features_kth('hog', params_hogf['params_hog'], n_patterns, list_actions)
        data_training_hof, data_validation_hof, data_testing_hof, label_training_hof, label_validation_hof, label_testing_hof = get_features_kth('hof', params_hogf['params_hof'], n_patterns, list_actions)
        data_training = np.hstack((data_training_hog, data_training_hof))
        data_validation = np.hstack((data_validation_hog, data_validation_hof))
        data_testing = np.hstack((data_testing_hog, data_testing_hof))
        assert((label_training_hog == label_training_hof).all())
        assert((label_validation_hog == label_validation_hof).all())
        assert((label_testing_hog == label_testing_hof).all())
        label_training = label_training_hog
        label_validation = label_validation_hog
        label_testing = label_testing_hog
    else:
        data_training, data_validation, data_testing, label_training, label_validation, label_testing = get_features_kth(descriptor, params_hogf, n_patterns, list_actions)

    #run_svm_sobel(data_training, data_validation, data_testing, label_training, label_validation, label_testing, type_svm)
    clf = svm.SVC(**params_svm).fit(np.vstack((data_training, data_validation)), np.hstack((label_training, label_validation)))
    pred = clf.predict(data_testing)
    pred = pred.reshape(pred.size/n_patterns, n_patterns)

    output = []
    for ans in pred:
       output.append(stats.mode(ans)[0])

    label_testing = label_testing.reshape(-1, n_patterns)[:, 0]
    accuracy = metrics.accuracy_score(label_testing, output)
    output = np.array(output)
    label_test = np.array(label_testing)
    print metrics.classification_report(output, label_testing)
    print accuracy
    if return_pred:
        return (output, label_testing, pred)
    return (output, label_testing)

sys.path.append(os.environ['GIT_REPO'] + '/source-code/optical-flow')
import opticalFlowAlgorithms
reload(opticalFlowAlgorithms)
def get_opticalFlow_kth(kth_info, params_filter_kth = (None, None, None), grid_step=12):
    import os
    global PATH_KTH_OPTICALFLOW
    global PATH_KTH
    for key, value in search_in_kth_info(kth_info, params_filter_kth):
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        file_path = '%s%s/person%02d_%s_d%d.npz' % (PATH_KTH_OPTICALFLOW, action, ith_person, action, ith_d)
        print file_path
        if os.path.exists(file_path): continue
        cap = cv2.VideoCapture('%s%s/person%02d_%s_d%d_uncomp.avi' % (PATH_KTH, action, ith_person, action, ith_d))
        mag, ang = opticalFlowAlgorithms.get_dense_opticalFlow(cap, dict(grid_step = grid_step))
        np.savez_compressed(file_path, mag, ang)

def get_patterns_opticalFlow(kth_info, params_filter_kth, save_patterns_as_img=False):
    import glob
    import re
    global PATH_KTH_PATTERNS
    global PATH_KTH_OPTICALFLOW
    global PATH_KTH_OPTICALFLOW_PATTERNS
    for key, value in search_in_kth_info(kth_info, params_filter_kth):
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        filename_pattern = glob.glob('%s%s/person%02d_%s_d%d*.png' % (PATH_KTH_PATTERNS, action, ith_person, action, ith_d))
        if not filename_pattern: continue
        pattern_opticalFlow = np.load('%s%s/person%02d_%s_d%d.npz' % (PATH_KTH_OPTICALFLOW, action, ith_person, action, ith_d))
        mag = pattern_opticalFlow['arr_0']
        ang = pattern_opticalFlow['arr_1']
        for path_pattern in filename_pattern:
            print path_pattern
            coord = re.split('/[\w]*', path_pattern[:-4])[-1]
            print coord
            posR, lenR, posC, lenC, posT, lenT = map(int, [x for x in re.split('_|\(|\)', coord) if x])
            pattern_mag = cv2.resize(mag[posT:posT+lenT, posR, posC:posC+lenC], (100, 100))
            pattern_ang = cv2.resize(ang[posT:posT+lenT, posR, posC:posC+lenC], (100, 100))
            pattern_sobel = cv2.imread(path_pattern, False) / 255.

            np.savez_compressed(path_pattern.replace(PATH_KTH_PATTERNS, PATH_KTH_OPTICALFLOW_PATTERNS)[:-4] + '.npz', [pattern_sobel * pattern_mag, pattern_sobel * pattern_ang])

            if save_patterns_as_img:
                hsv = np.empty([pattern_sobel.shape[0], pattern_sobel.shape[1], 3], np.uint8)
                hsv[..., 1] = 255
                hsv[..., 0] = pattern_sobel / 255. * pattern_ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(pattern_sobel / 255. * pattern_mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(path_pattern.replace(PATH_KTH_PATTERNS, PATH_KTH_OPTICALFLOW_PATTERNS), bgr)


def plot_confusion_matrix((y_test, y_pred), title='Confusion matrix', cmap=plt.cm.Blues):
    from sklearn import metrics
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(iris.target_names))
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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

def run_svm_sobel (data_training, data_validation, data_testing,
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

    print best_params
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

