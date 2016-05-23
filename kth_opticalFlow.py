import sys, os
import cv2
import numpy as np
import imutils
from pylab import plt
from sklearn.externals.joblib import Parallel, delayed
from imutils.object_detection import non_max_suppression
from subprocess import call, PIPE
from skimage import io

PATH_KTH = '/home/berthin/Documents/kth/'
PATH_KTH_OUT = '/home/berthin/Documents/kth-transformed/'
PATH_KTH_VR = '/home/berthin/Documents/kth-visual_rhythm-canny/'
PATH_KTH_PATTERNS = \
    '/home/berthin/Documents/kth-visual_rhythm-canny-patterns-morphology/'
KTH_CLASSES = ['boxing', 'handclapping', 'handwaving', 'walking']
KTH_CLASSES_INV = {KTH_CLASSES[idx]:idx for idx in xrange(len(KTH_CLASSES))}

"""
ith_class = 0
list_files = os.listdir(PATH_KTH + KTH_CLASSES[ith_class])
list_files.sort()
ext = '.avi'
list_files = [file for file in list_files if file[0] is not '.' and ext in file]
"""

#Parallel (n_jobs = 8) (delayed (run_denseOpticalFlowParallel) ((PATH_KTH + KTH_CLASSES[ith_class] + '/' + video_name), [0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [7], video_name]) for video_name in list_files)

#Parallel (n_jobs = 8) (delayed (show_denseOpticalFlow) (cv2.VideoCapture(PATH_KTH + KTH_CLASSES[ith_class] + '/' + video_name), 0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [7]) for video_name in list_files)

def empty_function (obj, params):
    return obj

def run_denseOpticalFlowParallel(video_path, params):
    cap = cv2.VideoCapture(video_path)
    show_denseOpticalFlow(cap, *params)

def show_denseOpticalFlow(cap, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags, thr, filter1, params1, filter2, params2, video_name=''):
    frame1 = cap.read()[1]
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    #while(1):
    for i in xrange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)):
        ret, frame2 = cap.read()
        if not ret: break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        next = filter1(next, *params1)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        print np.mean(hsv[..., 2][:])
        print np.max(hsv[..., 2][:])
        cv2.imshow(video_name,filter2(np.array(255*(hsv[..., 2] > thr),np.uint8),*params2))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

    cap.release()
    #cv2.destroyAllWindows()



#show_detectPeople((PATH_KTH + KTH_CLASSES[ith_class] + '/' + list_files[-1]), hog, list_files[-1])
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def show_detectPeople(video_path, video_name, winStride, padding, scale):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(video_path)
    for i in xrange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret: break
        orig = frame.copy()

    	# detect people in the image
    	(rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=scale)

    	# draw the original bounding boxes
    	for (x, y, w, h) in rects:
    		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    	# apply non-maxima suppression to the bounding boxes using a
    	# fairly large overlap threshold to try to maintain overlapping
    	# boxes that are still people
    	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    	# draw the final bounding boxes
    	for (xA, yA, xB, yB) in pick:
    		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow(video_name, frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

#does not work well
def show_detectPeople2(video_path, video_name, winStride, padding, scale, meanShift):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(video_path)
    for i in xrange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret: break
        orig = frame.copy()

    	# detect people in the image
    	(rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=scale, useMeanshiftGrouping=meanShift)

    	# draw the final bounding boxes
    	for (xA, yA, xB, yB) in rects:
    		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow(video_name, frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

#Parallel(n_jobs=6) (delayed(show_detectPeople)((PATH_KTH + KTH_CLASSES[ith_class] + '/' + file_name), hog, file_name) for file_name in list_files[:20])

#show_denseOpticalFlow(obj, 0.7, 3, 1, 5, 3, 1.2, 0, 10)
#show_denseOpticalFlow(obj, 0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [5])
#last tryshow_denseOpticalFlow(obj, 0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [7])

def read_boundingBox(file_name):
    global KTH_CLASSES_INV
    global PATH_KTH
    #person number (1-25) + scenario number (1-4) + action number (1-6) (1-boxing, 2-hand clapping, 3-hand waving, 4-jogging, 5-running, 6-walking)+ sequence number (1-4) + start frame + end frame +             bounding box information (ymin xmin ymax xmax) for each frame
    inFile = open(file_name, 'r')
    inFile.readline()
    inFile.readline()
    while True:
        seq1 = inFile.readline()
        seq2 = inFile.readline()
        seq3 = inFile.readline()
        seq4 = inFile.readline()
        arr = np.array([int(x) for x in seq1.split(' ') if x.isdigit()])
        video_name = 'person%02d_%s_d%d_uncomp.avi' % (arr[0], KTH_CLASSES[arr[2] - 1], arr[1])
        cap = cv2.VideoCapture(PATH_KTH + KTH_CLASSES[arr[2] - 1] + '/' + video_name)
        print cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for seq in [seq1, seq2, seq3, seq4]:
            print 'new seq'
            arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
            start_frame, end_frame = arr[4:6]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for ymin, xmin, ymax, xmax in np.hsplit(arr[6:], end_frame-start_frame+1):
                ret, frame = cap.read()
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.imshow(video_name, frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        break
    inFile.close()


def read_boundingBox_minmax(file_name, frame_size, padding = 10):
    global PATH_KTH
    global KTH_CLASSES
    #person number (1-25) + scenario number (1-4) + action number (1-6) (1-boxing, 2-hand clapping, 3-hand waving, 4-jogging, 5-running, 6-walking)+ sequence number (1-4) + start frame + end frame +             bounding box information (ymin xmin ymax xmax) for each frame
    inFile = open(file_name, 'r')
    inFile.readline()
    inFile.readline()
    while True:
        seq1 = inFile.readline()
        seq2 = inFile.readline()
        seq3 = inFile.readline()
        seq4 = inFile.readline()
        arr = np.array([int(x) for x in seq1.split(' ') if x.isdigit()])
        video_name = 'person%02d_%s_d%d_uncomp.avi' % (arr[0], KTH_CLASSES[arr[2] - 1], arr[1])
        cap = cv2.VideoCapture(PATH_KTH + KTH_CLASSES[arr[2] - 1] + '/' + video_name)
        print cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for seq in [seq1, seq2, seq3, seq4]:
            print 'new seq'
            arr = np.array([int(x) for x in seq.split(' ') if x.isdigit()])
            start_frame, end_frame = arr[4:6]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            bounds = arr[6:].reshape(end_frame - start_frame + 1, 4)
            ymin, xmin, ymax, xmax = [np.min(bounds[:, i]) for i in xrange(4)]

            ymin = max(ymin - padding, 0)
            xmin = max(xmin - padding, 0)
            ymax = min(ymax + padding, frame_size[0] - 1)
            xmax = max(xmax + padding, frame_size[1] - 1)

            for ith_frame in xrange(end_frame-start_frame+1):
                ret, frame = cap.read()
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.imshow(video_name, frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        break
    inFile.close()

def get_boundingBoxes(file_name, size, show=False):
    global KTH_CLASSES_INV
    global PATH_KTH
    global PATH_KTH_OUT
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

        while True:
            print video_name
            call(['nohup', 'mkdir', '/tmp/' + video_name, '>', '/dev/null', '2>1'])
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
                #cap_out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                cv2.imwrite('/tmp/' + video_name + '/' + str(idx_frame) + video_name[:-3] + 'png', frame)
                idx_frame += 1
                if show:
                    cv2.imshow(video_name, frame)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
            seq = inFile.readline()
            prefix = ' '.join(map(str, arr[:3]))
            if not seq.startswith(prefix):
                break
        call(['ffmpeg', '-i', '/tmp/'+video_name+'/'+'%d'+video_name[:-3]+'png', '-vcodec', 'mpeg4', '-y', PATH_KTH_OUT+KTH_CLASSES[arr[2] - 1] + '/' + video_name[:-3]+'mp4'])
        cap_in.release()
        print '----'
        #cap_out.release()
        #break
    inFile.close()


sys.path.append(os.environ['GIT_REPO'] + '/source-code/visual-rhythm')
import visual_rhythm

#read_kth_info(PATH_KTH + 'info-kth.in')
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

#search_in_kth_info(kth_info, (1, 'boxing', None))

def search_in_kth_info (m_kth_info, (ith_person, action, ith_d)):
    import re
    param_ith_person = '[0-9]+' if not ith_person else str(ith_person)
    param_action = '[a-z]+' if not action else action
    param_ith_d = '[0-9]+' if not ith_d else str(ith_d)
    return [(key, value) for key, value in m_kth_info.items() if re.search('_%s_%s_%s_' % (param_ith_person, param_action, param_ith_d), '_%s_' %'_'.join(map(str, key)))]

#according to an analysis, the min-max frame-interval is 68 among all the considered actions
def get_visualrhythm_improved_short_sequences(m_kth_info, n_frames=68, fraction = 1/4, type_visualrhythm = 'horizontal', params = None, frame_size = (120, 160), sigma_canny = 2.5):
    from skimage import feature
    global PATH_KTH
    global PATH_KTH_VR
    for key, value in m_kth_info.items():
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        cap = cv2.VideoCapture('%s%s/person%02d_%s_d%d_uncomp.avi' % (PATH_KTH, action, ith_person, action, ith_d))
        for start_frame, end_frame in value:
            diff = end_frame - start_frame + 1
            if diff < n_frames: continue
            img_vr = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + diff * fraction)
            for ith_frame in xrange(start_frame + diff * fraction, (end_frame - diff * fraction) + 1):
                _, frame = cap.read()
                img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params))
            img_vr = np.array(img_vr)
            img_vr = feature.canny(img_vr, sigma_canny) * 255
            cv2.imwrite('%s%s/person%02d_%s_d%d.png' % (PATH_KTH_VR, action, ith_person, action, ith_d), img_vr)
            break
        cap.release()

#test using all the sequence
def get_visualrhythm_improved_whole_sequences(m_kth_info, type_visualrhythm = 'horizontal', params = None, frame_size = (120, 160), sigma_canny = 2.5):
    from skimage import feature
    global PATH_KTH
    global PATH_KTH_VR
    for key, value in m_kth_info.items():
        ith_person, action, ith_d = key
        if len(value) == 0: continue
        #if not(ith_person == 6 and action == 'boxing' and ith_d == 1): continue
        #print 'person%02d_%s_d%d_uncomp.avi' % (ith_person, action, ith_d)
        cap = cv2.VideoCapture('%s%s/person%02d_%s_d%d_uncomp.avi' % (PATH_KTH, action, ith_person, action, ith_d))
        ith_p = 1
        for start_frame, end_frame in value:
            img_vr = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for ith_frame in xrange(start_frame, end_frame + 1):
                _, frame = cap.read()
                if frame is None: break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                img_vr.append(visual_rhythm.extract_from_frame(frame, type_visualrhythm, frame_size, params))
            if len(img_vr) < 10: break
            img_vr = np.array(img_vr)
            img_vr = feature.canny(img_vr, sigma_canny) * 255
            cv2.imwrite('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_VR, action, ith_person, action, ith_d, ith_p), img_vr)
            #for ir in xrange(0, img_vr.shape[1] - 160, 160):
            #   img_vr[:, ir:ir+160] = feature.canny(img_vr[:, ir:ir+160], 1.5)
            #cv2.imwrite('%s%s/__person%02d_%s_d%d_p%d.png' % (PATH_KTH_VR, action, ith_person, action, ith_d, ith_p), img_vr * 255)
            ith_p += 1
        cap.release()


def run_parallel_vr_extraction(path, action):
    kth_info = read_kth_info(path, list_actions = action)
    get_visualrhythm_improved_whole_sequences(kth_info, type_visualrhythm = 'horizontal', params = [5], frame_size = (120, 160), sigma_canny = 1.5)


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

            #print len(bag_patterns),
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

##if __name__ == '__main__':
    ##global PATH_KTH, KTH_CLASSES
    # import time
    # time1 = time.time()

    ##Parallel(n_jobs=-1)(delayed(run_parallel_vr_extraction)(PATH_KTH + 'info-kth.in', [action]) for action in KTH_CLASSES)
    #kth_info = read_kth_info (PATH_KTH + 'info-kth.in', list_actions = 'all')
    #get_visualrhythm_improved_whole_sequences(kth_info, type_visualrhythm = 'horizontal', params = [5], frame_size = (120, 160), sigma_canny = 1.5)

    # time2 = time.time()
    # print time2 - time1

    #import time
    #time1 = time.time()

    #Parallel(n_jobs=-1)(delayed(extract_patterns_smart_var)(action, frame_size=(120, 160), patt_size=(100,100), n_patterns=5, save_patterns=True) for action in KTH_CLASSES)

    #time2 = time.time()
    #print time2 - time1

from skimage.feature import hog
def hog_per_action(action = 'boxing', actors_training = [], params_hog = None, n_patterns_per_video = 0):
    global PATH_KTH_PATTERNS
    data = []
    for ith_actor in actors_training:
        for ith_d in xrange(1, 5):
            for ith_p in xrange(1, 1 + n_patterns_per_video):
                pattern = cv2.imread('%s%s/person%02d_%s_d%d_p%d.png' % (PATH_KTH_PATTERNS, action, ith_actor, action, ith_d, ith_p), False)
                if pattern is None: continue
                #pattern = (pattern > 0) * 255
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

