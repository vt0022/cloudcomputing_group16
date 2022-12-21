import os
from flask import Flask, flash, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import numpy as np
import cloudinary.uploader
from urllib.request import urlopen
from PIL import Image
import joblib

app = Flask(__name__)

app.static_folder = 'static'
filename = None

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLOUD_NAME = 'dx8k8cjdq'
API_KEY = '495478988912747'
API_SECRET = '90xd1xw4Ck3qlCaitFDUBMzM4e4'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/sign_language')
def sign_language():
    return render_template('sign_language_recognition.html')

@app.route('/sign_language_recogition', methods=['POST'])
def sign_language_recogition():
    if 'file' not in request.files:
        flash('Không tìm thấy file!')
        return render_template('sign_language_recognition.html')

    file = request.files['file']

    if file.filename == '':
        flash('Không có ảnh nào được tải lên!')
        return render_template('sign_language_recognition.html')

    if file and allowed_file(file.filename):
        # Upload ảnh lên cloudinary
        cloudinary.config(cloud_name = CLOUD_NAME, api_key=API_KEY, 
            api_secret=API_SECRET)
        upload_result = None

        upload_result = cloudinary.uploader.upload(file)

        filename = upload_result['url']

        loaded_best_model = keras.models.load_model("model_sign_language.h5")

        # Tên ký hiệu
        dict_labels = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i",9:"j"
        ,10:"k",11:"l",12:"m",13:"n",14:"o",15:"p",16:"q",17:"r",18:"s",19:"t",20:"u",
        21:"unkowen",22:"v",23:"w",24:"x",25:"y",26:"z"}


        # Load ảnh từ url
        img = Image.open(urlopen(filename))

        # Đổi cỡ ảnh
        img = img.resize((50, 50))

        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Bảng dự đoán phần trăm
        p = loaded_best_model.predict(img)

        # Số tương ứng
        label = np.argmax(p[0],axis=-1)
        name = dict_labels[label].upper()
        # Mức độ trùng khớp
        probality = np.max(p[0],axis=-1)
        
        return render_template('sign_language_recognition.html', image=filename, signname=name, probality=probality)
    else:
        flash('Định dạng ảnh hỗ trợ là png, jpg, jpeg, gif!')
        return render_template('sign_language_recognition.html')

if __name__ == "__main__":
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')

