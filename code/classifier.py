import numpy as np
import PExpert
import NExpert
import FeatureExtractor
from sklearn.svm import LinearSVC
import cv2
# from sklearn.datasets import make_classification

clf = LinearSVC(random_state=0, tol=1e-5, probability=True)

def fit_svc(frame, bbox, positive=True, init=True):
    positives = PExpert.generate_positives(frame, bbox, nparray=False)
    negatives = NExpert.generate_negative(frame, bbox, nparray=False)
    X = np.array(positives + negatives)
    y = [0] * (np.shape(positives)[0]) + [1] * (np.shape(negatives)[0])
    #print(np.shape(X))
    #print(np.shape(y))
    clf.fit(X, y)

def predict(frame, bbox):
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    pred = clf.predict(np.array([features]))
    print(pred)
    if pred[0] == 0:
        return True
    return False

def positive_score(frame, bbox):
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    return clf.score(np.array([features]), [0])

def negative_score(frame, bbox):
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    return clf.score(np.array([features]), [1])

if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    #bbox = cv2.selectROI(frame, False)
    bbox = (123, 14, 157, 180)
    #print(bbox)
    fit_svc(frame, bbox)
    print("Fit SVC done!")
    bbox = cv2.selectROI(frame, False)
    print(bbox)
    print(predict(frame, bbox))
    