import classifier
import cv2
import numpy as np

# detector应当是一个与tracker并行运行的程序

from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#video = cv2.VideoCapture("test_video.mp4")
#video = cv2.VideoCapture("ticao20190514_170124.mp4")
count = 0
sum_freq = 0


while count < 300:
    count = count + 1
    frame = cv2.imread("20180128_172532.jpg")
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    print("ok")
    #ok, frame = video.read()
    #if not ok:
    #    break

    timer = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    
    
    (rects, weights) = hog.detectMultiScale(frame,winStride=(4, 4),
		padding=(8, 8), scale=1.05)
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    sum_freq = sum_freq + fps

    for bbox in rects:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
    
    cv2.putText(frame, "People Detector (HOG, SVM)", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    cv2.putText(frame, "FPS : " + str(int(fps)) + " AVG : " + str(int(sum_freq / count)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1)


'''clf_hog = LinearSVC(random_state=0, tol=1e-5)
clf_color = LinearSVC(random_state=0, tol=1e-5)
clf = LinearSVC(random_state=0, tol=1e-5)

def SlidingWindow(frame, stepSize=5, windowSize=[150,150]):
    max_score = 0
    max_bbox = (0,0,0,0)
    for y in range(0, frame.shape[0], stepSize):
        if y + windowSize[1] >= frame.shape[0]:
            break
        for x in range(0, frame.shape[1], stepSize):
            if x + windowSize[0] >= frame.shape[1]:
                break
            #window = frame[y:y+windowSize[1], x:x+windowSize[0]]
            bbox = (x, y, windowSize[0], windowSize[1])
            score = classifier.positive_score(frame, bbox)
            if score == 1:
                return score, bbox
            print(score, bbox)
            if score > max_score:
                max_score = score
                max_bbox = bbox
    return max_score, max_bbox

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

def positive_score(frame, bbox, type="both"):
    if type == "hog":
        features = FeatureExtractor.extract_hog(frame, bbox)
        return clf_hog.score(np.array([features]), [1])
    elif type == "hsv":
        features = FeatureExtractor.extract_color(frame, bbox)
        return clf_color.score(np.array([features]), [1])
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    features = np.array([features])
    return clf_hog.score(features, [0]) * clf_color.score(features, [0])

def negative_score(frame, bbox, type="both"):
    if type == "hog":
        features = FeatureExtractor.extract_hog(frame, bbox)
        return clf_hog.score(np.array([features]), [1])
    elif type == "hsv":
        features = FeatureExtractor.extract_color(frame, bbox)
        return clf_color.score(np.array([features]), [1])
    features = FeatureExtractor.extract_hog_hsv(frame, bbox)
    return clf.score(np.array([features]), [1]) * clf_color.score(features, [1])

def updateClassifier(frame, bbox, threshold = 0.8):
    score = classifier.positive_score(frame, bbox)
    print(score)
    if score > threshold:
        # consider tracker is correct
        classifier.fit_svc(frame, bbox)
        return bbox
    elif score < 1 - threshold:
        # tracker failed
        # 全局搜索符合svm的窗口
        score, bbox = SlidingWindow(frame)
        return bbox

if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    #bbox = cv2.selectROI(frame, False)
    bbox = (123, 14, 157, 180)
    #print(bbox)
    classifier.fit_svc(frame, bbox)
    print("Fit SVC done!")
    #bbox = cv2.selectROI(frame, False)
    #newbox = updateClassifier(frame, bbox)
    score, newbox = SlidingWindow(frame)
    print(score, newbox)
    p1 = (int(newbox[0]), int(newbox[1]))
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)
'''