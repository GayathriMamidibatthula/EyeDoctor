import cv2
import numpy as np
import os
from IPython.display import Image,display
import re, os.path

def get_face_cascade(cascade='haarcascade_frontalface_default.xml'):
    return os.path.join(cv2.data.haarcascades, cascade)

def get_eye_cascade(cascade='haarcascade_eye.xml'):
    return os.path.join(cv2.data.haarcascades, cascade)


def start():
    eye_counter = 0

    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(get_face_cascade())
    eye_cascade = cv2.CascadeClassifier(get_eye_cascade())

    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true as long as the webcam is running
            print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w] 
                eyes = eye_cascade.detectMultiScale(roi_gray)
                #for (ex,ey,ew,eh) in eyes:
                # roi_color_eye = roi_color[ey:ey+eh, ex:ex+ew]
                # both_eyes = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                # cv2.imwrite(r"C:\Users\gayathri_sri_mamidib\Downloads\opencv-master\test\test\eye_%d.jpg" % eye_counter, roi_color_eye)
                # eye_counter += 1
                
        
            if key == ord('s'): 
                face_roi = cv2.resize(frame,(500,500))
                cv2.imwrite(filename= r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\uploaded\image\saved_img.jpg", img=face_roi)
                path = r'C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test'
                dir = os.listdir(path)
                if len(dir) != 0: 
                    mypath = r'C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test'
                    for root, dirs, files in os.walk(mypath):
                        for file in files:
                            os.remove(os.path.join(root, file))
        
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w] 
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex,ey,ew,eh) in eyes:
                        roi_color_eye = roi_color[ey:ey+eh, ex:ex+ew]
                        both_eyes = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        eyes_roi = cv2.resize(roi_color_eye,(150,150))
                        cv2.imwrite(r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test\eye_%d.jpg" % eye_counter, eyes_roi)
                        eye_counter += 1 
                        print("eyes saved")
                    
                webcam.release()
                img_new = cv2.imread(r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\uploaded\image\saved_img.jpg", cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread(r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\uploaded\image\saved_img.jpg", cv2.IMREAD_ANYCOLOR)
                print("TAKE PHOTO COMPLETE")
                break

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break        