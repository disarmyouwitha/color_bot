import os
import sys
import cv2
import numpy
import random
import imageio


# [HSV Calibrator]:
# https://piofthings.net/blog/opencv-baby-steps-4-building-a-hsv-calibrator

# [OpenCV Video from Webcam]:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# [Shape Detection]:
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

# [Text-to-Speech]:

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
    blue_ball_lower_hsv = numpy.array([95,136,50])
    blue_ball_upper_hsv = numpy.array([121,255,255])
    _blue_ball = (blue_ball_lower_hsv, blue_ball_upper_hsv)

    green_ball_lower_hsv = numpy.array([50,50,0])
    green_ball_upper_hsv = numpy.array([80,255,255])
    _green_ball = (green_ball_lower_hsv, green_ball_upper_hsv)

    yellow_ball_lower_hsv = numpy.array([19,128,0])
    yellow_ball_upper_hsv = numpy.array([34,255,255])
    _yellow_ball = (yellow_ball_lower_hsv, yellow_ball_upper_hsv)

    pink_ball_lower_hsv = numpy.array([166,118,130])
    pink_ball_upper_hsv = numpy.array([204,225,255])
    _pink_ball = (pink_ball_lower_hsv, pink_ball_upper_hsv)

    # [Create color_array]:
    _color_dict = {}
    _color_dict.update({'BLUE': _blue_ball})
    _color_dict.update({'GREEN': _green_ball})
    _color_dict.update({'YELLOW': _yellow_ball})
    _color_dict.update({'PINK': _pink_ball})

    return _color_dict

if __name__ == "__main__":
    # [Turn calibration on/off]:
    _CALIBRATE_HSV = False
    _CALIBRATE_THRESH = False

    # [Initialize HSV color values]:
    COLOR_DICT = init_color_presets()

    # [Get a random choice from COLOR_DICT]:
    _color_choice = random.choice(list(COLOR_DICT.keys()))
    (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]
    #print('Go and FETCH me a {0} ball!'.format(_color_choice))
    print('Go and FETCH me something.. {0}!'.format(_color_choice))

    # [Initialize Calibration Window]:
    if _CALIBRATE_HSV:
        calibration_window_name = 'HSV Calibrator'
        init_calibration_window(calibration_window_name, lower_hsv, upper_hsv)

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
        nemo = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)

         # [Threshold the HSV image]:
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # [Display the resulting frame]:
        #cv2.imshow(video_window_name, frame) # raw frame
        cv2.imshow(video_window_name, mask)

        # [Listen for ESC key]:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # [Check for MATCH]:
        _MASK_CNT = numpy.sum(mask == 255)
        _MASK_THRESH = 2000

        if _CALIBRATE_THRESH:
            print('_MASK_CNT({0}): {1}'.format(_color_choice, _MASK_CNT))

        if _MASK_CNT > _MASK_THRESH:
            #print('[Correct!]')
            print('[Correct!]: _MASK_CNT: {0}'.format(_MASK_CNT))
            #break

            if _CALIBRATE_HSV == False:
                # [Get a random choice from COLOR_DICT]:
                _color_choice = random.choice(list(COLOR_DICT.keys()))
                (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]
                #print('Now, Go and FETCH me a {0} ball!'.format(_color_choice))
                print('Now, Go and FETCH me something.. {0}!'.format(_color_choice))

        if _CALIBRATE_HSV:
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