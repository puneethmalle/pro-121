# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480
mountain.set(640,480)

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])
        upper_bound = np.array([255,255,255])
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        # thresholding image
        mountain = cv2.morphologyEx(mountain, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mountain = cv2.morphologyEx(mountain, cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
        # inverting the mask

        # bitwise and operation to extract foreground / person
        mask = cv2.bitwise_not(mountain)
        result1 = cv2.bitwise_and(frame,frame,mask = mask)
        result2 = cv2.bitwise_and(frame,frame,mask = mountain)
        # final image
        final_output = cv2.addWeighted(result1,1,result2,1,0)
        cv2.imshow("video",final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
