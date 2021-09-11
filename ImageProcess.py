import cv2
import os
import os.path



def get_face_cascade(cascade='haarcascade_frontalface_default.xml'):
    return os.path.join(cv2.data.haarcascades, cascade)

def get_eye_cascade(cascade='haarcascade_eye.xml'):
    return os.path.join(cv2.data.haarcascades, cascade)

def mainfunc(filename):
    eye_counter = 0
    face_cascade = cv2.CascadeClassifier(get_face_cascade())
    eye_cascade = cv2.CascadeClassifier(get_eye_cascade())

    img = cv2.imread(os.path.join('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample/uploaded/image', filename))
    #img = cv2.imread('C:/Users/Abigail_Gracias/Documents/Sample/uploaded/image/image.jpg')
    print("IMAGE READ")

    #make picture gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
    face_roi = cv2.resize(img,(500,500))
    cv2.imwrite(filename= r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\uploaded\image\saved_img.jpg", img=face_roi)
    path = r'C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test'
    dir = os.listdir(path)
    if len(dir) != 0: 
        mypath = r'C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test'
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            roi_color_eye = roi_color[ey:ey+eh, ex:ex+ew]
            both_eyes = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyes_roi = cv2.resize(roi_color_eye,(150,150))
            cv2.imwrite(r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Sample\model\test\test\eye_%d.jpg" % eye_counter, eyes_roi)
            eye_counter += 1 
            print("eyes saved")
    
#cv2.imshow('my image',face_roi)
#cv2.imsave('my image',eye1)
#cv2.imwrite(filename= r"C:\Users\gayathri_sri_mamidib\Desktop\Hackathons\Eye Doctor\Eye and CNN\saved_img.jpg", img=face_roi)

#cv2.waitKey(1650)
#cv2.destroyAllWindows()
#print("Processing image...")
#img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
#print("Converting RGB image to grayscale...")
#gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
#print("Converted RGB image to grayscale...")
#print("Resizing image to 150x150 scale...")
#img_ = cv2.resize(gray,(500,500))
#print("Resized...")
#img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
#print("Image saved!")
