import cv2
import numpy as np
import random
import FeatureExtractor

def inBox(point,box):
    if point[0] >= box[0] and point[0] <= box[0] + box[2] and point[1] >= box[1] and point[1] <= box[1] + box[3]:
        return True
    return False

def generate_negative(frame, bbox, sample_size=20, max_iteration=1000, feature=True, nparray=True):
    height, width, channels = frame.shape
    iteration = 0
    negatives = []
    while True:
        if len(negatives) >= sample_size:
            break
        point1 = (int(random.random() * height), int(random.random() * width))
        if inBox(point1, bbox):
            iteration = iteration + 1
            if(iteration > max_iteration):
                break
            continue
        while(True):
            point2 = (int(random.random() * height), int(random.random() * width))
            if inBox(point2, bbox) or abs(point1[0] - point2[0]) < 20 or abs(point1[1] - point2[1]) < 20:
                iteration = iteration + 1
                if(iteration > max_iteration):
                    break
                continue
            '''sign1 = np.random.randint(0,2)
            if sign1 == 0:
                sign1 = -1
            sign2 = np.random.randint(0,2)
            if sign2 == 0:
                sign2 = -1
            point2 = (point1[0]+h*sign1, point2[0]+w*sign2)'''
            #point3 = (point1[0], point2[1])
            #point4 = (point2[0], point1[1])
            #newbox = (point1[0], point1[1], bbox[2], bbox[3])
            newbox = (
                min(point1[1], point2[1]),
                min(point1[0], point2[0]),
                abs(point1[1] - point2[1]),
                abs(point1[0] - point2[0]))
            if  (newbox[0] > bbox[0]+bbox[2] or newbox[0]+newbox[2] < bbox[0] or newbox[1] > bbox[1] + bbox[3] or newbox[1]+newbox[3] < bbox[1]):
                #print(newbox)
                #ROI = frame[newbox[1]:newbox[1]+newbox[3],newbox[0]:newbox[0]+newbox[2]]
                #height, width, channels = ROI.shape
                #if(height > 10 and width > 10):
                if feature:
                    negatives.append(FeatureExtractor.extract_hog_hsv(frame, newbox))
                else:
                    negatives.append(newbox)
                if len(negatives) >= sample_size:
                    if nparray:
                        return np.array(negatives)
                    else:
                        return negatives
                break
            iteration = iteration + 1
            if(iteration > max_iteration):
                break
    if nparray:
        return np.array(negatives)
    else:
        return negatives


if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    bbox = cv2.selectROI(frame, False)
    print(bbox)
    print("negatives: " + str(np.shape(generate_negative(frame, bbox))))
    negatives = generate_negative(frame, bbox, feature=False)
    for newbox in negatives:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)
    '''scale = bboxScale(frame, bbox)
    for newbox in scale:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)'''
    '''roi = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    roi = cv2.flip(roi, 1)
    cv2.imshow("image",roi)
    cv2.waitKey(0)'''