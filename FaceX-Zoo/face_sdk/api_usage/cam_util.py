import sys
import time
import os

sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

if __name__ == '__main__':
    # common setting for all models, need not modify.
    model_path = 'models'

    # face detection model setting.
    scene = 'mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face recognition model setting.
    #scene = 'model-at'
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face recognition model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # read image and get face features.
    #image_path = 'api_usage/test_images/test1.jpg'
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    face_cropper = FaceRecImageCropper()
    path = 'C:/user/cjs/k_celeb/'
    file_list = os.listdir(path)
    feature_test = []
    for i in range(len(file_list)):
        save_path = 'C:/user/cjs/feature_set/' + file_list[i] + '.npy'
        feature_test.append(np.load(save_path))
    print(len(feature_test))

    prev_time = 0
    #FPS = 100
    #video_file = 'C:/user/cjs/test_video/office.mp4'
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if cap.isOpened():
        while True:
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            current_time = time.time()
            sec = current_time - prev_time
            prev_time = current_time
            fps = 1 / (sec)
            strs = "FPS : %0.1f" % fps
            if ret:
                prev_time = time.time()
                dets = faceDetModelHandler.inference_on_image(image)
                face_nums = dets.shape[0]
                feature_list = []
                for i in range(face_nums):
                    landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
                    landmarks_list = []
                    for (x, y) in landmarks.astype(np.int32):
                        landmarks_list.extend((x, y))
                    cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
                    #cv2.imwrite('api_usage/temp/my_no_mask_cropped_test.jpg', cropped_image)
                    feature = faceRecModelHandler.inference_on_image(cropped_image)
                    #print(feature)
                    feature_list.append(feature)
                    name = 'no-matching'
                    for j in range(len(file_list)):
                        score = np.dot(feature_list[i], feature_test[j])
                        if score > 0.5:
                            print(score)
                            name = file_list[j][0:len(file_list[j])-4]
                            break
                    cv2.rectangle(image, (int(dets[i][0]), int(dets[i][1])), (int(dets[i][2]), int(dets[i][3])), (0, 0, 255), 2)
                    cv2.putText(
                        image,
                        name,
                        (int(dets[i][0]), int(dets[i][1]) - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 512, (255, 0, 0))
                    cv2.putText(image, strs, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                cv2.imshow('camera', image)
                if cv2.waitKey(1) != -1:
                    break
            else:
                print('no frame')
                break
    else:
        print("can't open camera.")
    cap.release()
    cv2.destroyAllWindows()