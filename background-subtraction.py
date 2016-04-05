# tmp

# coding: utf-8

# In[1]:

import cv2
import os
from sklearn.externals.joblib import Parallel, delayed


# In[44]:

cd ~/Documents/kth


# In[45]:

def empty_function(obj, *params): return obj


# In[42]:

def run(video_name, params_MOG, filter_prev, filter_prev_params, filter_post, filter_post_params):
    sub = cv2.bgsegm.createBackgroundSubtractorMOG()
    sub.setNMixtures(params_MOG['NMixtures'])
    sub.setBackgroundRatio(params_MOG['BackgroundRatio'])
    sub.setNoiseSigma(params_MOG['NoiseSigma'])
    sub.setHistory(params_MOG['History'])
    cap = cv2.VideoCapture(video_name)
    while True:
        ret, frame = cap.read()
        if not ret: break
        mask = sub.apply(filter_prev(frame, *filter_prev_params))
        cv2.imshow('video_name', filter_post(mask, *filter_post_params))
        cv2.waitKey(30)
    cv2.destroyAllWindows()


# In[17]:

run('boxing/person01_boxing_d4_uncomp.avi', {'NMixtures':5, 'BackgroundRatio':0.3})


# In[4]:

list_files = os.listdir('boxing')


# In[47]:

sub_params = {'NMixtures':5, 'BackgroundRatio':0.3, 'NoiseSigma':10, 'History':250}
filter_prev, filter_prev_params = cv2.medianBlur, [3]
filter_post, filter_post_params = empty_function, [None]
action = 'boxing'
list_files = [video_name for video_name in os.listdir(action) if video_name[0] is not '.' and video_name[-3:] == 'avi']
_ = Parallel (n_jobs = 8) (delayed (run) (action+'/'+video_name, sub_params, filter_prev, filter_prev_params, filter_post, filter_post_params) for video_name in list_files[:20])


# In[34]:

sub = cv2.bgsegm.createBackgroundSubtractorMOG()
print sub.getNMixtures(), sub.getBackgroundRatio(), sub.getNoiseSigma(), sub.getHistory()


# In[33]:

help(sub)


# In[ ]:



