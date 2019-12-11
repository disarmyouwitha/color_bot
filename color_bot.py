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

# [HSV Calibrator]:
# https://piofthings.net/blog/opencv-baby-steps-4-building-a-hsv-calibrator

# [OpenCV Video from Webcam]:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# [Shape Detection]:
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

# [Text-to-Speech]:
# os.system("say Hello World")

def shape_detection(masked_frame=None):
    # [For testing frame can be set to an image]:
    if masked_frame is None:
        frame = cv2.imread('shapes_and_colors.png')
        #frame = cv2.imread('correct.png')
    else:
        frame = masked_frame

    # [Load the image and resize it to a smaller factor so that the shapes can be approximated better]:
    resized = imutils.resize(frame, width=300)
    ratio = frame.shape[0] / float(resized.shape[0])
    # CAN RESIZE HAPPEN AFTER BLUR?

    if masked_frame is None:
        # [Convert the resized image to grayscale, blur it slightly, and threshold it]:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    else: #FRAME SHOULD BE SENT WITH MASK
        thresh = frame

    # [Find contours in the thresholded image and initialize the shape detector]:
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = shape_detector.shape_detector()

    print('_NUM_CONTOURS: {0}'.format(cnts))

    _cnt = 0
    # [Loop over the contours]:
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        print('_SHAPE: {0}'.format(shape))

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2) # CHANGE COLOR OF BOUNDING BOX HERE?
        cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        _cnt += 1

        # show the output image
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

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


# [OSX Text-To-Speach]:
def say(text, BLOCKING=False):
    print(text)
    if BLOCKING:
        p = subprocess.call(['say','{0}'.format(text)])
    else:
        p = subprocess.Popen(['say','{0}'.format(text)])


# [Need to do TTS in a different thread than WHILE loop]:
# [If we go above threshold.. record for ~5sec and return the frame with the highest _MASK_CNT]:
# ^(Use this for shape detection)
if __name__ == "__main__":
    # [Turn calibration on/off]:
    _CALIBRATE_SHAPE = True
    _CALIBRATE_HSV = False # 'YELLOW'
    _SCREEN_SHOTS = False

    # [Detect color]:
    if _CALIBRATE_SHAPE == False:
        # [Initialize HSV color values]:
        COLOR_DICT = init_color_presets()

        # [Initialize Calibration Window]:
        if _CALIBRATE_HSV == False:
            # [Get a random choice from COLOR_DICT]:
            _color_choice = random.choice(list(COLOR_DICT.keys()))
            (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]

            # [Text to Speech on OSX]:
            say('Go and FETCH me something.. {0}!'.format(_color_choice), BLOCKING=True)
        else:
            calibration_window_name = 'HSV Calibrator'
            (lower_hsv, upper_hsv) = COLOR_DICT[_CALIBRATE_HSV]
            init_calibration_window(calibration_window_name, lower_hsv, upper_hsv)
            

            # [Text to Speech on OSX]:
            say('Go and FETCH me something.. {0}!'.format(_CALIBRATE_HSV), BLOCKING=True)

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

            # [Set MATCH Threshold]:
            _MASK_CNT = numpy.sum(mask == 255)
            _MASK_THRESH = 3000
            #_MASK_MIN_CONTOURS = 3

            # [Check MATCH Conditions]:
            if _MASK_CNT > _MASK_THRESH:
                if _CALIBRATE_HSV != False: # Testing
                    print('_MASK_CNT: {0}'.format(_MASK_CNT))
                    # PRINT NUMBER OF COUNTOURS?
                else:
                    say('Correct!', BLOCKING=True)

                # [Screenshot user playing game / having fun!]:
                if _SCREEN_SHOTS:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imageio.imwrite('correct_{0}.png'.format(_ss_cnt), rgb_frame)
                    _ss_cnt+=1

                # [Shape Detection]: (might need to move this up to check countours for correct status later)
                shape_detection(mask)

                #print('[Sleeping 3 sec]..')
                #time.sleep(3)

                if _CALIBRATE_HSV == False:
                    # [Get a random choice from COLOR_DICT]:
                    _color_choice = random.choice(list(COLOR_DICT.keys()))
                    (lower_hsv, upper_hsv) = COLOR_DICT[_color_choice]

                    # [Text to Speech on OSX]:
                    say('Now, Go and FETCH me something.. {0}!'.format(_color_choice), BLOCKING=True)

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

    # [Detect Shape]:
    if _CALIBRATE_SHAPE == True:
        shape_detection()