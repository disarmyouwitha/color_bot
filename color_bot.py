import os
import sys
import cv2
import time
import numpy
import random
import imageio
import imutils

# [HSV Calibrator]:
# https://piofthings.net/blog/opencv-baby-steps-4-building-a-hsv-calibrator

# [OpenCV Video from Webcam]:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# [Shape Detection]:
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

# [Text-to-Speech]:
# os.system("say Hello World")

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
        shape = "circle"

        # return the name of the shape
        return shape

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


# [Need to do TTS in a different thread than WHILE loop]:
# [If we go above threshold.. record for ~5sec and return the frame with the highest _MASK_CNT]:
# ^(Use this for shape detection)
if __name__ == "__main__":
    # [Detect Shape]:

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread('shapes_and_colors.png')
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    '''
    # [Turn calibration on/off]:
    _CALIBRATE_HSV = False
    _CALIBRATE_THRESH = False

    # [Initialize HSV color values]:
    COLOR_DICT = init_color_presets()

    # [Get a random choice from COLOR_DICT]:
    _color_choice = random.choice(list(COLOR_DICT.keys()))
    (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]

    # [Text to Speech on OSX]:
    text = 'Go and FETCH me something.. {0}!'.format(_color_choice)
    print(text)
    os.system('say {0}'.format(text))

    # [Initialize Calibration Window]:
    if _CALIBRATE_HSV:
        calibration_window_name = 'HSV Calibrator'
        init_calibration_window(calibration_window_name, lower_hsv, upper_hsv)

    # [Initialize video capture source]: (macbook webcam)
    cap = cv2.VideoCapture(0)

    video_window_name = 'Computer Vision'
    cv2.namedWindow(video_window_name)
    cv2.moveWindow(video_window_name, 200,300)

    _ss_cnt = 0
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

        if _MASK_CNT > _MASK_THRESH:
            if _CALIBRATE_THRESH:
                print('[Correct!]: _MASK_CNT: {0}'.format(_MASK_CNT))
            else:
                print('[Correct!]')

            # [Screenshot user playing game / having fun!]:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageio.imwrite('correct_{0}.png'.format(_ss_cnt), rgb_frame)
            _ss_cnt+=1

            print('[Sleeping 3 sec]..')
            time.sleep(3)
            #break

            if _CALIBRATE_HSV == False:
                # [Get a random choice from COLOR_DICT]:
                _color_choice = random.choice(list(COLOR_DICT.keys()))
                (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]

                # [Text to Speech on OSX]:
                text = 'Now, Go and FETCH me something.. {0}!'.format(_color_choice)
                print(text)
                os.system('say {0}'.format(text))

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
    '''