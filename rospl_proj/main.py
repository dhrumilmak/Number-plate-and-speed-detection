from turtle import tracer
from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract
import math
import time



#line 
bx1=20
by=400
bx2=225



pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")


cap = cv2.VideoCapture('vid.mp4')


def Speed_Cal(time):
   
    try:
        Speed = (9.14*3600)/(time*1000)               
        return Speed
    except ZeroDivisionError:
        print (5)
        
#car num

start_time = time.time()



# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

   

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nplate = cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in nplate:
                wT, hT, cT = frame.shape
                a, b = (int(0.02*wT), int(0.02*hT))
                plate = frame[y+a:y+h-a, x+b:x+w-b, :]

                # make the frame more darker to identify LPR
                kernel = np.ones((1, 1), np.uint8)
                plate = cv2.dilate(plate, kernel, iterations=1)
                plate = cv2.erode(plate, kernel, iterations=1)
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                (thresh, plate) = cv2.threshold(
                    plate_gray, 55, 200, cv2.THRESH_BINARY)

                # cv2.imshow('Frame', plate)



            
                cv2.line(frame,(bx1,by),(bx2,by),(255,0,0),2)
                while int(by) <= int((y+y+h)/2)&int(by+10) >= int((y+y+h)/2):
                    cv2.line(frame,(bx1,by),(bx2,by),(0,255,0),2)
                    Speed = Speed_Cal(time.time() - start_time)
                    print(" Speed: "+str(Speed))
                   
                    cv2.putText(frame, "Speed: "+str(Speed)+"KM/H", (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3);
                    break



                # read the text on the plate
                read = pytesseract.image_to_string(plate)

                read = ''.join(e for e in read if e.isalnum())
                stat = read[0:2]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 51, 255), 2)
                cv2.rectangle(frame, (x-1, y-40), (x+w+1, y), (51, 51, 255), -1)
                cv2.putText(frame, read, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            
        # Display the resulting frameq
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
            if cv2.waitKey() & 0xFF == ord('q'):           
                break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
