import numpy as np
import PExpert
import NExpert
import FeatureExtractor
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import cv2
import imutils
from sklearn import svm
import slidingwindow as sw
# from sklearn.datasets import make_classification

#clf = LinearSVC(random_state=0, tol=1e-5)
clf = svm.SVC(kernel="rbf",  C=1.0, probability=True)
bdt = AdaBoostClassifier(n_estimators=100, random_state=0)

def fit_adaboost(frame, bbox, plural=False):
    timer = cv2.getTickCount()
    if plural:
        positives = []
        negatives = []
        for i in range(len(frame)):
            #print(np.shape(frame[i]), bbox[i])
            positives= positives + PExpert.generate_positives(frame[i], bbox[i], nparray=False)
            negatives = negatives + NExpert.generate_negative(frame[i], bbox[i], nparray=False)
    else:
        positives = PExpert.generate_positives(frame, bbox, nparray=False)
        negatives = NExpert.generate_negative(frame, bbox, nparray=False)
    X = np.array(positives + negatives)
    y = [0] * (np.shape(positives)[0]) + [1] * (np.shape(negatives)[0])
    #print("Fitting...")
    bdt.fit(X, y)
    #print("Score: ", bdt.score(X,y))
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    print("FIT ADABOOST: size: ", np.shape(X), "FPS: ",fps)
    '''
    for i in range(50, len(positives), 50):
        p_slice = positives[:i]
        n_slice = negatives[:i]
        X_slice = np.array(p_slice+n_slice)
        y_slice = [0] * (np.shape(p_slice)[0]) + [1] * (np.shape(n_slice)[0])
        timer = cv2.getTickCount()
        bdt = AdaBoostClassifier(n_estimators=100, random_state=0)
        bdt.fit(X_slice, y_slice)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        #print("FIT SVM: size: ", np.shape(X_slice), "FPS: ",fps)
        print(np.shape(X_slice)[0],fps)'''

def SlidingWindow(frame, overlap=0.8, windowSize=[150,150], classifier="svm", scale=1.5):
    max_score = 0
    max_bbox = (0,0,0,0)
    #sz_x = [int(windowSize[0] / scale / scale), int(windowSize[0] / scale), windowSize[0], int(windowSize[0] * scale), int(windowSize[0] * scale * scale)]
    windowSize[0] = round(windowSize[0] / scale )
    windowSize[1] = round(windowSize[1] / scale )
    for i in range(3):
        step_x = round(windowSize[0] * (1 - overlap))
        step_y = round(windowSize[1] * (1 - overlap))
        for y in range(0, frame.shape[0], step_y):
            if y + windowSize[1] >= frame.shape[0]:
                break
            for x in range(0, frame.shape[1], step_x):
                if x + windowSize[0] >= frame.shape[1]:
                    break
                #window = frame[y:y+windowSize[1], x:x+windowSize[0]]
                bbox = (x, y, windowSize[0], windowSize[1])
                score = positive_score(frame, bbox, classifier=classifier)
                if score == 1:
                #if predict(frame, bbox, classifier):
                    return 1, bbox
                #print(score, bbox)
                if score > max_score:
                    max_score = score
                    max_bbox = bbox
        windowSize[0] = round(windowSize[0] * scale)
        windowSize[1] = round(windowSize[1] * scale)
    return max_score, max_bbox

def fit_svc(frame, bbox, positive=True, init=True, plural=False, neg_sample=10):
    if plural:
        positives = []
        negatives = []
        for i in range(len(frame)):
            try:
                positives= positives + PExpert.generate_positives(frame[i], bbox[i], nparray=False)
            except:
                print("something wrong with pos bbox")
            try:
                negatives = negatives + NExpert.generate_negative(frame[i], bbox[i], nparray=False)
            except:
                print("something wrong with neg bbox")
    else:
        positives = PExpert.generate_positives(frame, bbox, nparray=False)
        negatives = NExpert.generate_negative(frame, bbox, nparray=False, max_iteration=100*neg_sample)
    X = np.array(positives + negatives)
    y = [0] * (np.shape(positives)[0]) + [1] * (np.shape(negatives)[0])
    #print(np.shape(X))
    #print(np.shape(y))
    timer = cv2.getTickCount()
    clf.fit(X, y)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    print("FIT SVM: size: ", np.shape(X), "FPS: ",fps)
    '''
    for i in range(50, len(positives), 50):
        p_slice = positives[:i]
        n_slice = negatives[:i]
        X_slice = np.array(p_slice+n_slice)
        y_slice = [0] * (np.shape(p_slice)[0]) + [1] * (np.shape(n_slice)[0])
        timer = cv2.getTickCount()
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(X_slice, y_slice)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        #print("FIT SVM: size: ", np.shape(X_slice), "FPS: ",fps)
        print(np.shape(X_slice)[0],fps)'''

def predict(frame, bbox, classifier="svm"):
    bbox = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    if classifier == 'adaboost':
        pred = bdt.predict(np.array([features]))
    else:
        pred = clf.predict(np.array([features]))
    #print(pred)
    if pred[0] == 0:
        return True
    return False

def positive_score(frame, bbox, classifier='svm'):
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    if classifier == 'adaboost':
        return bdt.score(np.array([features]), [0])
    #return clf.score(np.array([features]), [0]
    score = clf.predict_proba([features])
    #print(score)
    return score[0][0]

def negative_score(frame, bbox, classifier='svm'):
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    if classifier == 'adaboost':
        return bdt.score(np.array([features]), [1])
    return clf.score(np.array([features]), [1])

if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    #bbox = cv2.selectROI(frame, False)
    bbox = (123, 14, 157, 180)
    #print(bbox)
    fit_svc(frame, bbox)
    fit_adaboost(frame, bbox)
    print("Fit SVC done!")
    bbox = cv2.selectROI(frame, False)
    print(bbox)
    print(predict(frame, bbox))
    score, newbox = SlidingWindow(frame)
    print(score, newbox)
    p1 = (int(newbox[0]), int(newbox[1]))
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)
    #print(cv2.getTickFrequency())

    '''video = cv2.VideoCapture(0)
    ok, frame = video.read()
    frame = imutils.resize(frame, width=min(frame.shape[1], 500))
    print(frame.shape)
    count = 0
    timer = cv2.getTickCount()
    init_width = frame.shape[1]
    init_height = frame.shape[0]
    while count < 1:
        s, newbox = SlidingWindow(frame,stepSize=min(30,int(0.3*init_width), int(0.3*init_height)), windowSize=[150,150], classifier="svm")
        #print(s)
        #p1 = (int(newbox[0]), int(newbox[1]))
        #p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #cv2.imshow("image",frame)
        #cv2.waitKey(0)
        count = count + 1
    print((cv2.getTickCount() - timer)/cv2.getTickFrequency())
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #print("Sliding Window: FPS: ",fps)'''