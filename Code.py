from flask import Flask, render_template, request 
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator 
import tensorflow as tf 
import numpy as np 
import os
import ImageProcess
import testeye
import eye


#model = tf.keras.models.load_model('model') 
app = Flask(__name__) 

app.config['UPLOAD_FOLDER'] = 'uploaded/image'

@app.route('/')
def home():
    try: 
        import shutil
        shutil.rmtree('uploaded') 
        os.mkdir('uploaded')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample/uploaded')
        os.mkdir('image')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample/model/test')
        shutil.rmtree('test')
        os.mkdir('test')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample')
        #% cd uploaded % mkdir image % cd .. 
        print("folder created") 
    except: 
        print("folder not created")
        pass
    return render_template('eyedoctorUI.html')


@app.route('/home') 
def upload_i(): 
    try: 
        import shutil
        shutil.rmtree('uploaded') 
        os.mkdir('uploaded')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample/uploaded')
        os.mkdir('image')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample/model/test')
        shutil.rmtree('test')
        os.mkdir('test')
        os.chdir('C:/Users/gayathri_sri_mamidib/Desktop/Hackathons/Eye Doctor/Sample')
        #% cd uploaded % mkdir image % cd .. 
        print("folder created") 
    except: 
        print("folder not created")
        pass
    return render_template('upload.html') 

'''def finds(): 
	test_datagen = ImageDataGenerator(rescale = 1./255) 
	vals = ['Cat', 'Dog'] # change this according to what you've trained your model to do 
	test_dir = 'uploaded'
	test_generator = test_datagen.flow_from_directory( 
			test_dir, 
			target_size =(224, 224), 
			color_mode ="rgb", 
			shuffle = False, 
			class_mode ='categorical', 
			batch_size = 1) 

	pred = model.predict_generator(test_generator) 
	print(pred) 
	return str(vals[np.argmax(pred)])'''

@app.route('/uploader', methods = ['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        f = request.files['file'] 
        #secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        print("completed")
        ImageProcess.mainfunc(f.filename)
        val = testeye.mainfunc() 
        return render_template('result.html', e1 = val[0], e2 = val[1]) 

@app.route('/takephoto', methods = ['GET', 'POST'])
def take_photo():
    eye.start()
    val = testeye.mainfunc()
    return render_template('result.html', e1 = val[0], e2 = val[1]) 

if __name__ == '__main__': 
	app.run() 