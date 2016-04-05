import sys, os
import cv2
import numpy as np
import imutils
from pylab import plt
from sklearn.externals.joblib import Parallel, delayed
from imutils.object_detection import non_max_suppression
from subprocess import call, PIPE

PATH_KTH = '/home/berthin/Documents/kth/'
PATH_KTH_OUT = '/home/berthin/Documents/kth-transformed/'
PATH_KTH_VR = '/home/berthin/Documents/kth-transformed-visual_rhythm/'
KTH_CLASSES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running' , 'walking']
KTH_CLASSES_INV = {KTH_CLASSES[idx]:idx for idx in xrange(len(KTH_CLASSES))}

ith_class = 0
list_files = os.listdir(PATH_KTH + KTH_CLASSES[ith_class])
list_files.sort()
ext = '.avi'
list_files = [file for file in list_files if file[0] is not '.' and ext in file]

Parallel (n_jobs = 8) (delayed (run_denseOpticalFlowParallel) ((PATH_KTH + KTH_CLASSES[ith_class] + '/' + video_name), [0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [7], video_name]) for video_name in list_files)

Parallel (n_jobs = 8) (delayed (show_denseOpticalFlow) (cv2.VideoCapture(PATH_KTH + KTH_CLASSES[ith_class] + '/' + video_name), 0.7, 3, 1, 5, 3, 1.2, 0, 10, empty_function, [None], cv2.medianBlur, [7]) for video_name in list_files)

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
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
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

Parallel(n_jobs=6) (delayed(show_detectPeople)((PATH_KTH + KTH_CLASSES[ith_class] + '/' + file_name), hog, file_name) for file_name in list_files[:20])

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

def get_visualrhythm(file_name, size, type_visualrhythm = 'horizontal', params = None, show = False):
    def extract_horizontal(img, (W, H), gap):
        return np.array([img[row][:] for row in xrange(0, H, gap)], np.uint8).flatten()
    def extract_vertical(img, (W, H), gap):
        return np.array([img[:][col] for col in xrange(0, W, gap)], np.uint8).flatten()
    def extract_zigzag(img, (W, H), row_gap, col_gap):
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

