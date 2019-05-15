import classifier
import cv2
import numpy as np

# detector应当是一个与tracker并行运行的程序

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
            print(score, bbox)
            if score > max_score:
                max_score = score
                max_bbox = bbox
    return max_score, max_bbox

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
