import cv2
import dlib
import numpy as np

class face_key_point_marker():
    def __init__(self,model_path="shape_predictor_68_face_landmarks.dat"):
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(model_path)

    def get_key_point(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        rects=self.detector(img_gray,0)
        if(len(rects)==0):
            return None
        landmark_matrix=[]
        for i in range(0,len(rects)):
            landmarks=np.matrix([[p.x,p.y] for p in self.predictor(img,rects[i]).parts()])
            landmark_matrix.append(landmarks)
        return landmark_matrix

    def write_on_image(self,img,landmark_matrix,point_size=5):
        print(img.dtype)
        print(img.shape)
        for i in range(0,len(landmark_matrix)):
            landmarks=landmark_matrix[i]
            for idx,point in enumerate(landmarks):
                pos=(point[0,0],point[0,1])
                cv2.circle(img,pos,point_size,color=(255,255,255),thickness=-1)
        return img

    def write_feature_map(self,landmark_matrix,image_shape,point_size=5):
        img=np.zeros((image_shape[0],image_shape[1],1)).astype(np.float32)
        img=self.write_on_image(img,landmark_matrix)
        return img

marker=face_key_point_marker()
img=cv2.imread("test.jpg")

landmark_matrix=marker.get_key_point(img)
image=marker.write_on_image(img,landmark_matrix)
feature_map=marker.write_feature_map(landmark_matrix,img.shape)
