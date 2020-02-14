import cv2
import numpy as np

cap = cv2.VideoCapture('car1.mp4')

frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 640 resolution
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))     #360 resolution
 
print(frame_width )

fourcc = cv2.VideoWriter_fourcc('X','V','I','D') # other method (*'xvid') # fourcc is codec video for output/writing of video

out = cv2.VideoWriter("output.avi",fourcc, 5.0, (1280,720))


ret, frame1 = cap.read()        # frame1 is GUI over back video,
ret, frame2 = cap.read()        # frame2 is vedio feed in back side

print(frame1.shape)     # video array pixel shape

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)      # To find absolute frame difference
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)         #(5,5 ) is kernel value , 0 is sigma X value(increase value cause loss in accuracy ex: 0.1,0.5,or 1000)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) 
    # _ parameter(ret = boolean value) we r not using # 20 thresh value (value decrease's more unnecessary and vice versa, 255 is max thres value,#type is thresh_bin
    
    dilated = cv2.dilate(thresh, None, iterations=3)                # kernel size is None
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # _ hierachy parameter which  r not using
    # Hierachy is the optional out vector which is containing the information about image topology.

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)    # create cnts using boundingRect

        if cv2.contourArea(contour) < 900:          # create cnt if its under area(900)
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) #(frame1,co-ordinates,increment of area, (3 color values) , thickness of cnts)    # green cnts over all car are created here.
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3) #
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    #line = cv2.line(image, (0,255), (255,255), (147, 96, 44), 50)
    out.write(image)
    cv2.imshow("feed_Title", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:       # use ESc. for exit
        break
    ''' if cv2.waitKey(1) & 0xFF == ord('q'):   # TRY THIS 'q' is quit , and 0xFF for 64bit system.
        break
    '''

cv2.destroyAllWindows()
cap.release()                   # Release cache of both cap & out  { cap.read,cap.VedioCapture and out.write }
out.release()