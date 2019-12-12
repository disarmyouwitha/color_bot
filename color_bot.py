import os
import sys
import cv2
import time
import numpy
import random
import imageio
import imutils
import subprocess
import shape_detector
from threading import Thread


def shape_detection(thresh, _SHAPE_THRESH=500):
    # [Resize image to a smaller factor so that the shapes can be approximated better]:
    resized = imutils.resize(thresh, width=300)
    ratio = thresh.shape[0] / float(resized.shape[0])

    # [Find contours in the thresholded image and initialize the shape detector]:
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # [Loop over the getting rid of the small ones]:
    _contours = []
    for c in cnts:
        A = cv2.contourArea(c) # find area of contour
        if A > _SHAPE_THRESH:
            _contours.append(c)

    # [Initialize shape detector]:
    sd = shape_detector.shape_detector()

    # [Loop over the contours]:
    _shapes = []
    for c in _contours:
        # [Compute the center of the contour, then detect the name of the shape using only the contour]:
        M = cv2.moments(c)
        A = cv2.contourArea(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)

        #print('_NUM_CONTOURS(thresh/total): {0}/{1}'.format(len(_contours), len(cnts)))
        #print('_SHAPE: {0} | _AREA: {1}'.format(shape, A))
        _shapes.append((shape, A))

        # [Multiply the contour (x, y)-coordinates by the resize ratio, then draw the contours and the name of the shape on the image]:
        #c = c.astype("float")
        #c *= ratio
        #c = c.astype("int")
        #cv2.drawContours(frame, [c], -1, (0, 255, 0), 2) # CHANGE COLOR OF BOUNDING BOX HERE?
        #cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # ^(I want to still use this to draw the contours it finds)

        # [Show the output image]:
        #cv2.imshow("shape_frame", frame)
        #cv2.waitKey(0)

    return _shapes

def nothing(self, x=''):
    #print('Trackbar value: ' + str(x))
    pass

def init_calibration_window(calibration_window_name, lower_hsv, upper_hsv):
    # [Unpack into local variables]:
    (uh, us, uv) = upper_hsv
    (lh, ls, lv) = lower_hsv

    # [Set up HSV Calibration window]:
    cv2.namedWindow(calibration_window_name)
    cv2.moveWindow(calibration_window_name, 900,10) 

    # [Create trackbars for Hue]:
    cv2.createTrackbar('LowerH',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerH',calibration_window_name, lh)
    cv2.createTrackbar('UpperH',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperH',calibration_window_name, uh)

    # [Create trackbars for Saturation]:
    cv2.createTrackbar('LowerS',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerS',calibration_window_name, ls)
    cv2.createTrackbar('UpperS',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperS',calibration_window_name, us)

    # [Create trackbars for Value]:
    cv2.createTrackbar('LowerV',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerV',calibration_window_name, lv)
    cv2.createTrackbar('UpperV',calibration_window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperV',calibration_window_name, uv)

    font = cv2.FONT_HERSHEY_SIMPLEX

def init_color_presets():
    # [Threshold Presets]:
    blue_lower_hsv = numpy.array([95,136,50])
    blue_upper_hsv = numpy.array([121,255,255])
    _blue = (blue_lower_hsv, blue_upper_hsv)

    green_lower_hsv = numpy.array([50,50,0])
    green_upper_hsv = numpy.array([80,255,255])
    _green = (green_lower_hsv, green_upper_hsv)

    yellow_lower_hsv = numpy.array([19,160,80])
    yellow_upper_hsv = numpy.array([34,255,255])
    _yellow = (yellow_lower_hsv, yellow_upper_hsv)

    pink_lower_hsv = numpy.array([166,118,130])
    pink_upper_hsv = numpy.array([204,225,255])
    _pink = (pink_lower_hsv, pink_upper_hsv)

    purple_lower_hsv = numpy.array([128,80,20])
    purple_upper_hsv = numpy.array([145,255,255])
    _purple = (purple_lower_hsv, purple_upper_hsv)

    red_lower_hsv = numpy.array([0,80,160])
    red_upper_hsv = numpy.array([10,255,255])
    _red = (red_lower_hsv, red_upper_hsv)

    # [Create color_array]:
    _color_dict = {}
    _color_dict.update({'BLUE': _blue})
    _color_dict.update({'GREEN': _green})
    _color_dict.update({'YELLOW': _yellow})
    _color_dict.update({'PINK': _pink})
    _color_dict.update({'PURPLE': _purple})
    _color_dict.update({'RED': _red})

    return _color_dict


# [OSX Text-To-Speach]:
def text_to_speech(_say, _text=None, BLOCKING=False):
    if _text is None:
        _text = _say

    print(_text)
    if BLOCKING:
        p = subprocess.call(['say','{0}'.format(_say)])
    else:
        p = subprocess.Popen(['say','{0}'.format(_say)])

_ss_cnt = 0
def threaded_picture(frame):
    global _ss_cnt
    _rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imageio.imwrite('correct_{0}.png'.format(_ss_cnt), _rgb_frame)
    _ss_cnt+=1
    print('[Picture Taken]')

# Def need to thread the screenshot.
if __name__ == "__main__":
    _CALIBRATE_HSV = False #'PINK'
    _SCREEN_SHOTS = True

    _CORRECT_ELAPSED = 0
    _CORRECT_START = 0

    # [Detect color]:
    # [Initialize HSV color values]:
    COLOR_DICT = init_color_presets()

    # [Initialize Calibration Window]:
    if _CALIBRATE_HSV == False:
        # [Get a random choice from COLOR_DICT]:
        _color_choice = random.choice(list(COLOR_DICT.keys()))
        (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]

        # [Text to Speech on OSX]:
        text_to_speech('Go and FETCH me something.. {0}!'.format(_color_choice), BLOCKING=True)
    else:
        calibration_window_name = 'HSV Calibrator'
        (lower_hsv, upper_hsv) = COLOR_DICT[_CALIBRATE_HSV]
        init_calibration_window(calibration_window_name, lower_hsv, upper_hsv)
        text_to_speech('Go and FETCH me something.. {0}!'.format(_CALIBRATE_HSV), BLOCKING=True)

    # [Initialize video capture source]: (macbook webcam)
    cap = cv2.VideoCapture(0)

    video_window_name = 'Computer Vision'
    cv2.namedWindow(video_window_name)
    cv2.moveWindow(video_window_name, 200,300)

    while(True):
        # [Capture frame-by-frame]:
        ret, frame = cap.read()

        # [Median Blur]:
        # [Convert BGR to HSV]:
        # [Threshold the HSV image]:
        nemo = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        # [Display the resulting frame]:
        cv2.imshow(video_window_name, mask)

        # [Listen for ESC key]:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # [Set MATCH Threshold]:
        _MASK_CNT = numpy.sum(mask == 255)
        _MASK_THRESH = 1000

        # [Check MATCH Conditions]:
        if _MASK_CNT > _MASK_THRESH:
            # [Send masked image to Shape Detection]: (to filter out small artifacts in threshold)
            _SHAPES = shape_detection(mask, _SHAPE_THRESH=2000)

            # [If more than 1 shape found, correct!]:
            if(len(_SHAPES)>=1):
                if _CORRECT_START == 0:
                    print('[LOCKING ON]: (Hold for 2 seconds!)')
                    _CORRECT_START = time.time()
                    #print('_CORRECT_START: {0}'.format(_CORRECT_START))
                    

                if _CORRECT_ELAPSED >= 2: # If _CORRECT for more than 2 seconds:
                    
                    (_shape, _area) = _SHAPES[0]

                    if _CALIBRATE_HSV == False:
                        text_to_speech('Correct!', 'Correct! ({0})'.format(_shape), BLOCKING=True)
                    else:
                        text_to_speech('{0}!'.format(_shape), BLOCKING=False) #True

                    # [Screenshot user playing game / having fun!]: (Send to thread?)
                    if _SCREEN_SHOTS:
                        #_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #imageio.imwrite('correct_{0}.png'.format(_ss_cnt), _rgb_frame)

                        thread = Thread(target=threaded_picture, args=(frame, ))
                        thread.start()
                        # ^(Definitely need to thread this still)

                    #break

                    if _CALIBRATE_HSV == False:
                        # [Get a random choice from COLOR_DICT]:
                        _color_choice = random.choice(list(COLOR_DICT.keys()))
                        (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]
                        text_to_speech('Now, Go and FETCH me something.. {0}!'.format(_color_choice), BLOCKING=True)

                    #print('_CORRECT_ELAPSED(clear): {0}'.format(_CORRECT_ELAPSED))
                    _CORRECT_START = 0
                    _CORRECT_ELAPSED = 0
                else:
                    _CORRECT_ELAPSED = (time.time() - _CORRECT_START)
                    #print('_CORRECT_ELAPSED: {0}'.format(_CORRECT_ELAPSED))
            else:
                if _CORRECT_START > 0:
                    print('[Lost Focus]: Try again!')
                    _CORRECT_START = 0
                    _CORRECT_ELAPSED = 0

        if _CALIBRATE_HSV != False:
            # [Get current positions of Upper HSV trackbars]:
            uh = cv2.getTrackbarPos('UpperH',calibration_window_name)
            us = cv2.getTrackbarPos('UpperS',calibration_window_name)
            uv = cv2.getTrackbarPos('UpperV',calibration_window_name)
            # [Get current positions of Lower HSCV trackbars]:
            lh = cv2.getTrackbarPos('LowerH',calibration_window_name)
            ls = cv2.getTrackbarPos('LowerS',calibration_window_name)
            lv = cv2.getTrackbarPos('LowerV',calibration_window_name)
            # [Set lower/upper HSV to get the current mask]:
            upper_hsv = numpy.array([uh,us,uv])
            lower_hsv = numpy.array([lh,ls,lv])

    # [When everything done, release the capture]:
    cap.release()
    cv2.destroyAllWindows()
    print('[fin.]')
