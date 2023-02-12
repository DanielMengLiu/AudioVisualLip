"""
source:
https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector3(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, device=self.device, min_face_size=40)

    def detect_from_numpy(self, numpy_img):
        return self._detect_from_numpy(numpy_img)

    def _detect_from_numpy(self, numpy_img):
        image = Image.fromarray(numpy_img)
        boxes, prob = self.model.detect(image)
        prob = prob.tolist()
        # pil image bbox : left, upper, right, lower
        # cv2 image bbxo : left, upper, width, hight 
        boxes = boxes.tolist()
        for idx, bbox in enumerate(boxes):
            new_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])]
            boxes[idx] = new_bbox
        # cut roi images
        rois = []
        for bbox in boxes:
            roi = numpy_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            rois.append(roi)
        return boxes, prob, rois


class FaceEmbeder(object):
    # img to vec
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_from_numpy(self, numpy_img):
        return self._detect_from_numpy(numpy_img)

    def _detect_from_numpy(self, numpy_img):
        image = Image.fromarray(numpy_img)
        image = transforms.ToTensor()(image).to(self.device).unsqueeze(0)
        embedding = self.model(image).detach().cpu().tolist()[0]
        return embedding
        

class FaceDistanceCalculator(object):
    def calculate_distance(self, vec1, vec2, mode='cosine'):
        return self._calculate_distance(vec1, vec2, mode)

    def _calculate_distance(self, vec1, vec2, mode):
        if mode == 'euclidean':
            distance = np.linalg.norm(np.array(vec1)-np.array(vec2)).tolist()
        elif mode == 'cosine':
            distance = np.dot(np.array(vec1), np.array(vec2))/(np.linalg.norm(np.array(vec1))*np.linalg.norm(np.array(vec2)))
            distance = np.absolute(distance)
            distance = distance.tolist()
        return distance


class AgeAndSexPredictor(object):
    def __init__(self, 
                    age_proto=str(Path(__file__).parent / 'age_and_sex_model/age_deploy.prototxt'),
                    age_model=str(Path(__file__).parent / 'age_and_sex_model/age_net.caffemodel'), 
                    gender_proto=str(Path(__file__).parent / 'age_and_sex_model/gender_deploy.prototxt'),
                    gender_model=str(Path(__file__).parent / 'age_and_sex_model/gender_net.caffemodel')
                ):
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        
        self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def predict(self, np_image):
        return self._predict(np_image)

    def _predict(self, np_image):
        blob = cv2.dnn.blobFromImage(np_image, 1.0, (227, 227), self.model_mean_values, swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        return gender, age


class IdCardCheckerFullSystem(object):
    def __init__(self):
        self.face_detector = FaceDetector3()
        self.face_embeder = FaceEmbeder()
        self.distance_calulator = FaceDistanceCalculator()
        self.sex_and_age_predictor = AgeAndSexPredictor()

        self.distance_treshold = 0.6
        self.face_treshold = 0.85
        self.face_image_size = (100, 100)

    def predict_images_as_np(self, id_card_image, selffe_image):
        return self._predict_images_as_np(id_card_image, selffe_image)

    def _predict_images_as_np(self, id_card_image, selffe_image):
        output_dict = {}
        boxes, probs, rois = self.face_detector.detect_from_numpy(id_card_image)
        boxes, probs, id_card_face_images = self._drop_not_valid_boxes_rois(boxes, probs, rois)
        # i take only first most likly image from idcard
        id_card_face_image = id_card_face_images[0]
        output_dict.update(
            {
                'id_card_face_image': id_card_face_image,
                'id_card_face_prob': probs[0],
            }
        )
        boxes, probs, rois = self.face_detector.detect_from_numpy(selffe_image)
        boxes, probs, selfee_face_images = self._drop_not_valid_boxes_rois(boxes, probs, rois)
        # i take only first and second most likly images from selffe
        selfee_face_images = selfee_face_images[:2]
        output_dict.update({
                'selfee_face_images': selfee_face_images,
                'selfee_face_prob': probs[:2],
            })
        id_card_face_gender, id_card_face_age = self.sex_and_age_predictor.predict(id_card_face_image)
        output_dict.update(
            {
                'id_card_face_gender': id_card_face_gender,
                'id_card_face_age': id_card_face_age,
            })
        selffe_genders, selffe_ages = [], []
        for selfee_face_image in selfee_face_images:
            selfee_face_gender, selfee_face_age = self.sex_and_age_predictor.predict(selfee_face_image)
            selffe_genders.append(selfee_face_gender)
            selffe_ages.append(selfee_face_age)
        output_dict.update({
                'selffe_genders': selffe_genders,
                'selffe_ages': selffe_ages,
            })
        id_card_face_embeding = self.face_embeder.detect_from_numpy(id_card_face_image)
        selfee_face_embedings = []
        for selfee_face_image in selfee_face_images:
            selfee_face_embedings.append(self.face_embeder.detect_from_numpy(selfee_face_image))
        output_dict.update({
                'id_card_face_embeding': id_card_face_embeding,
                'selfee_face_embedings': selfee_face_embedings,
            })
        id_card_face_and_selfee_face1_distance = self.distance_calulator.calculate_distance(id_card_face_embeding, selfee_face_embedings[0])
        id_card_face_and_selfee_face2_distance = self.distance_calulator.calculate_distance(id_card_face_embeding, selfee_face_embedings[1])
        selfee_face1_selfee_face2_distance = self.distance_calulator.calculate_distance(selfee_face_embedings[0], selfee_face_embedings[1])
        output_dict.update({
                'id_card_face_and_selfee_face1_distance': id_card_face_and_selfee_face1_distance,
                'id_card_face_and_selfee_face2_distance': id_card_face_and_selfee_face2_distance,
                'selfee_face1_selfee_face2_distance': selfee_face1_selfee_face2_distance,
            })
        if id_card_face_and_selfee_face1_distance > self.distance_treshold:
            output_dict.update({'accepted': True})
        else:
            output_dict.update({'accepted': False})           

        return output_dict

    def _drop_not_valid_boxes_rois(self, boxes, probs, rois):
        new_boxes = []
        new_prob = []
        new_rois = []
        for box, prob, roi in zip(boxes, probs, rois):
            if prob > self.face_treshold:
                new_boxes.append(box)
                new_prob.append(prob)
                roi = cv2.resize(roi, self.face_image_size)
                new_rois.append(roi)
        return new_boxes, new_prob, new_rois
        

if __name__ == '__main__':
    id_card_image_path = '/datasets2/voxceleb1/unzippedIntervalFaces/data/A.J._Buckley/1.6/1zcIwhmdeo4/1/05.jpg'
    id_card_image = cv2.imread(id_card_image_path)
    # id_card_image = cv2.cvtColor(id_card_image, cv2.COLOR_BGR2RGB)
    # id_card_image = cv2.rotate(id_card_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width, channels = id_card_image.shape
    dsize = (width, height)

    # selffe_image_path = '/datasets2/voxceleb1/unzippedIntervalFaces/data/A.J._Buckley/1.6/1zcIwhmdeo4/1/01.jpg'
    # selffe_image = cv2.imread(selffe_image_path)
    # selffe_image = cv2.cvtColor(selffe_image, cv2.COLOR_BGR2RGB)
    # selffe_image = cv2.resize(selffe_image, dsize)

    # face_embeder = FaceEmbeder()
    # embedding0 = face_embeder.detect_from_numpy(id_card_image)
    # embedding1 = face_embeder.detect_from_numpy(selffe_image)

    # face_distance_calculator = FaceDistanceCalculator()
    # distance = face_distance_calculator.calculate_distance(embedding0, embedding1)
    # print(distance)

    # korwin 0.9
    # kornel 0.87 0.85 0.9
    # morawieki 0.9 0.6 0.7
    # dominik 0.57
    # kornel - domink 0.8 0.1 0.6
    # kornel


    # from pprint import pprint
    # full_system = IdCardCheckerFullSystem()
    # output = full_system.predict_images_as_np(id_card_image, selffe_image)
    # # pprint(output)
    # print(output['accepted'])
    # print('id_card_face_and_selfee_face1_distance:', output['id_card_face_and_selfee_face1_distance'])
    # print('id_card_face_and_selfee_face2_distance:', output['id_card_face_and_selfee_face2_distance'])
    # print('selfee_face1_selfee_face2_distance:', output['selfee_face1_selfee_face2_distance'])

    # for idx, roi in enumerate(output['selfee_face_images']):
    #     img_path = str(idx)+'.jpg'
    #     cv2.imwrite(img_path, roi)

    # img_path = 'idcard_face.jpg'
    # cv2.imwrite(img_path, output['id_card_face_image'])

    face_detector = FaceDetector3()
    boxes, prob, rois = face_detector.detect_from_numpy(id_card_image)

    # face_embeder = FaceEmbeder()
    # embeddings = []
    # for idx, roi in enumerate(rois):
    #     img_path = str(idx)+'.jpg'
    #     cv2.imwrite(img_path, roi)
    #     embedding = face_embeder.detect_from_numpy(roi)
    #     embeddings.append(embedding)

    # face_distance_calculator = FaceDistanceCalculator()
    # distance = face_distance_calculator.calculate_distance(embedding[0], embedding[0])
    # print(distance)

    sex_and_age_predictor = AgeAndSexPredictor()
    for idx, roi in enumerate(rois):
        gender, age = sex_and_age_predictor.predict(roi)
        print(gender, age)
    
    # skonczenie 
    # zrobnie datasetu z 10 osob

    # selffe_image_path = 'kobieta/2.jpg'
    # selffe_image = cv2.imread(selffe_image_path)
    # # selffe_image = cv2.cvtColor(selffe_image, cv2.COLOR_BGR2RGB)
    # face_detector = FaceDetector3()
    # boxes, prob, rois = face_detector.detect_from_numpy(selffe_image)
    # for idx, roi in enumerate(rois):
    #     img_path = str(idx)+'.jpg'
    #     cv2.imwrite(selffe_image_path, roi)