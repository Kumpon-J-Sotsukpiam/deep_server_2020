from app.config import APP_ROOT, SERVER_ROOT
from app.config import DATATEMP_ROOT, DATAMODEL_ROOT, DATASET_ROOT, DATAUPLOAD_ROOT
from app.config import IMAGE_SIZE
from flask import Flask, request
from app.model import VGGModel
from app.keras_vggface.utils import preprocess_input
import time
import os
import io
from io import StringIO, BytesIO
import base64
import datetime
import cv2
import json
import sys
import threading
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras import backend as k
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
from app.datafolder import cloneDir, precessData, normalization, copyDataProcess, delTree
from app.labels import writeJson, readJson
import keras.backend.tensorflow_backend as tb
#graph = tf.get_default_graph()
app = Flask(__name__)


def thread_function():
        tb._SYMBOLIC_SCOPE.value = True
    #global graph
    #with graph.as_default():
        while True:
            time.sleep(10)
            if(len(TrainList) > 0):
                try:
                    # simple early stopping
                    temp_name = TrainList[0]
                    temp_path = os.path.join(DATATEMP_ROOT, temp_name)
                    model_path = os.path.join(DATAMODEL_ROOT, temp_name+".h5")
                    label_path = os.path.join(DATAMODEL_ROOT, temp_name)
                    # before process
                    dataSetList = findStudentListbyClass(temp_name)
                    # processing data
                    cloneDir(DATASET_ROOT, DATATEMP_ROOT,
                             temp_name, dataSetList)
                    dataProcessed = precessData(DATASET_ROOT, dataSetList, 0.2)
                    normalization(dataProcessed)
                    copyDataProcess(dataProcessed, DATASET_ROOT, temp_path)
                    # build model
                    model = VGGModel()
                    model.build_model(nb_classes=len(dataSetList))
                    # train model
                    model.train(temp_path, nb_epoch=50)
                    # save model
                    model.save(model_path)
                    np.save(label_path, model.labels)
                    delTree(temp_path)
                    TrainList.pop(0)
                except Exception as err:
                    print(err)
                    delTree(temp_path)
                    # TrainList.pop(0)


def thread_detection():
    while True:
        time.sleep(10)
        if(len(DetecList) > 0):
            path_dataSet = DetecList[0]['path_dataSet']
            file_path = DetecList[0]['file_path']
            cap = cv2.VideoCapture(file_path)  # load file video
            face_list = []  # for save math of faces
            while True:
                ret, frame = cap.read()  # read frame from video
                if not ret:  # for protect last frame if not ret break loop
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade_classifier.detectMultiScale(
                    gray, minNeighbors=3, scaleFactor=1.3, minSize=(224, 224))
                if (len(faces) >= 1):
                    for (x, y, w, h) in faces:
                        cv2.rectangle(gray, (x, y), (x+w, y+h),
                                      (255, 255, 0), 2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)
                        # eye detection if not continue frame else save and continue
                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in eyes:
                            print(ex, ey, ew, eh)
                        face_list.append(roi_color)
            for i, face in enumerate(face_list):
                path = os.path.join(path_dataSet, str(i)+".jpg")
                print(path)
                cv2.imwrite(path, face)
            # end test code
            DetecList.pop(0)


def main_thread():
    while True:
        time.sleep(8)
        print(modelList)
        print(TrainList)
        print(DetecList)


# ================== Don't Change Anything =======================
modelList = {}
TrainList = ['5e102e79454c0a4d28ee4b67']
DetecList = []
# ========================= End ===================================
# ==================    Thread Function    ========================
threading.Thread(target=thread_function).start()
threading.Thread(target=thread_detection).start()
threading.Thread(target=main_thread).start()
# ========================= End ===================================
cascade_classifier = cv2.CascadeClassifier(
    APP_ROOT+'/libs/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(APP_ROOT+'/libs/haarcascade_eye.xml')
mongoClient = MongoClient()
db_sa = mongoClient['sa']

switcher = {
    -1: "Can't find a model",
    0: "Queue Train Model",
    1: "Training Model",
    2: 'Have Model'
}


def findStudentListbyClass(_id):
    ObjectIdList = []
    x = db_sa['classes'].find({'_id': ObjectId(_id)})
    for y in x[0]['studentList']:
        ObjectIdList.append(str(y))
    return ObjectIdList


def checkStatusDataSetList(list):
    error = []
    for x in list:
        y = os.path.join(DATASET_ROOT, x)
        if len(os.listdir(y)) == 0:
            error.append(x)
        else:
            print(y)
    return error


def checkStatusDataSet(_id):
    y = os.path.join(DATASET_ROOT, _id)
    status = -1
    if os.path.isdir(y) and len(os.listdir(y)) > 0:
        status = 1
    else:
        for x in DetecList:
            print(x['_id'])
            if x['_id'] == _id:
                status = 0
    return {'status': status}


def checkStatusModel(_id):
    x = os.listdir(DATAMODEL_ROOT)
    status = -1
    if(_id+".h5" in x):
        status = 2
    if _id in TrainList:
        if TrainList.index(_id) == 0:
            status = 1
        else:
            status = 0
    return {'status': status}


def norm_frame(frame=None, s=3):
    w, h, c = frame.shape
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(w / s), int(h / s)),
                       interpolation=cv2.INTER_LINEAR)
    return frame


def readVideo_faceDetection(path_video, path_dataSet):
    cap = cv2.VideoCapture(path_video)
    face_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = norm_frame(frame)
        face = faceDetection(frame)
        if face != []:
            face_list.append(face)
    for i, face in enumerate(face_list):
        path = os.path.join(path_dataSet, str(i)+".jpg")
        print(path)
        cv2.imwrite(path, face)
    return len(face_list)

# def deCodeBase64ToImage(b64):
#     imgdata = base64.b64decode(str(b64))
#     image = Image.open(io.BytesIO(imgdata))
#     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     return image

def faceDetection(image):
    gray_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        gray_data, minNeighbors=5, minSize=(150, 150))
    image = []
    if (len(faces) >= 1):
        for (x, y, w, h) in faces:
            cv2.rectangle(gray_data, (x, y), (x+w, y+h), (255, 255, 0), 2)
            roi_color = gray_data[y:y+h, x:x+w]
            roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            image.append(roi_color)
    return image


def prepareInput(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32, copy=False)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image, version=2)
    return image


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello World ! "


@app.route('/data_set/video', methods=['POST'])
def data_set_video():
    data = request.form
    file_name = data["name"]
    file_path = os.path.join(DATAUPLOAD_ROOT, file_name)
    path_dataSet = os.path.join(DATASET_ROOT, file_name[0:24])
    if not os.path.exists(path_dataSet):
        os.mkdir(path_dataSet)
    DetecList.append({
        "path_dataSet": path_dataSet,
        "file_path": file_path,
        "_id": file_name[0:24]
    })
    return {"ok": True, "message": "Query Face Detection"}

# append class_id to queue trainList
@app.route('/model', methods=['POST'])
def array():
    data = request.form
    _id = data["_id"]
    listOfClassId = findStudentListbyClass(_id)
    dataRes = {"ok": False, "message": "Not have Student in class"}, 500
    if len(listOfClassId) > 0:
        errors = checkStatusDataSetList(listOfClassId)
        if(len(errors) == 0):
            TrainList.append(_id)
            status = 1 if TrainList.index(_id) == 0 else 0
            dataRes = {"ok": True, "message": switcher.get(
                status), "status": status}, 200
        else:
            dataRes = {"ok": False, "message": errors}, 500
    return dataRes

# pop class_id from queue trainList
@app.route('/model/pop', methods=['DELETE'])
def popQueue():
    data = request.form
    _id = data["_id"]
    index = TrainList.index(_id)
    if index == 0:
        return {"ok": False, "error": "training model"}
    else:
        TrainList.pop(index)
        return {"ok": True, "status": checkStatusModel(_id)}


@app.route('/predict', methods=['POST'])
def predict():
    #global graph
    #with graph.as_default():
        data = request.form
        print(data)
        _uid = data['_uid']
        checkId = data['checkId']
        file = request.files['file']
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        image = cv2.imdecode(data, color_image_flag)
        image = faceDetection(image)
        dataRes = {"ok": False, "predicted": []}
        try:
            if image != []:
                for face in image:
                    face = prepareInput(face)
                    preds = modelList[_uid]['model'].predict(face)
                    predicted_class = np.argmax(preds, axis=1)[0]
                    # preds_min = np.min(preds)
                    preds_max = np.max(preds)
                    if preds_max > 0.95:
                        predict_label = modelList[_uid]['label'][predicted_class]
                        if(predict_label not in modelList[_uid]['checkIn']):
                            dataRes['ok'] = True
                            ObjectAppend = {
                                "_id": str(predict_label),
                                "time": datetime.datetime.now(),
                                "type": 'face'
                            }
                            dataRes['predicted'].append(ObjectAppend)
                            modelList[_uid]['checkIn'].append(predict_label)
            if(dataRes['ok']):
                for x in dataRes['predicted']:
                    db_sa['checkins'].update_one({'_id': ObjectId(checkId)}, {
                        '$addToSet': {'studentList': x}})
        except Exception as err:
            print(err)
        return dataRes


@app.route('/model', methods=['GET'])  # pass
def setmodel():
    #global graph
    #with graph.as_default():
        data = request.form
        _id = data["_id"]
        _uid = data["_uid"]
        path_model = os.path.join(DATAMODEL_ROOT, _id+".h5")
        path_label = os.path.join(DATAMODEL_ROOT, _id+".npy")
        if _uid in modelList and modelList[_uid]['_id'] == _id:
            return {"ok": True, "message": "Loaded Model", "status": 1}, 200
        elif _uid in modelList:
            del modelList[_uid]
        status = checkStatusModel(_id)['status']
        if status == -1:  # no model.h5
            return {"ok": False, "message": switcher.get(status), "status": status}, 404
        elif status == 0 or status == 1:  # training model
            return {"ok": False, "message": switcher.get(status), "status": status}, 404
        else:  # 1 # have model.h5
            model = VGGModel()
            check, message = model.load(path_model)
            labels = np.load(path_label)
            modelList[_uid] = {"_id": _id, "model": model,
                               "label": labels, 'checkIn': []}
        return {"ok": check, "message": message, "status": status}, 200 if check else 404


@app.route('/model', methods=['DELETE'])
def delmodel():
    data = request.form
    _id = data["_id"]
    _uid = data["_uid"]
    if _uid in modelList:
        del modelList[_uid]
    return {"ok": True}


@app.route('/model/check', methods=['GET'])
def checkModel():
    data = request.form
    _id = data['_id']
    check = checkStatusModel(_id)
    check['message'] = switcher.get(check['status'])
    return check, 200


@app.route('/data_set/check', methods=['GET'])
def checkDataSet():
    data = request.form
    _id = data['_id']
    return checkStatusDataSet(_id)
