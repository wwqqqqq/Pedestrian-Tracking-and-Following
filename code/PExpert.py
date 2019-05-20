import FeatureExtractor
import numpy as np
import cv2

def bboxShift(frame, bbox, max_shift=0.08):
    height, width, channels = frame.shape
    max_h = int(max_shift * bbox[3])
    max_w = int(max_shift * bbox[2])
    if max_h <  2:
        max_h = 2
    if max_w < 2:
        max_w = 2
    uplimit = min(height - (bbox[1] + bbox[3]), max_h)
    downlimit = min(bbox[1], max_h)
    leftlimit = min(bbox[0], max_w)
    rightlimit = min(width - (bbox[0] + bbox[2]), max_w)
    samples = []
    # left
    if leftlimit > 1:
        step = np.random.randint(1, leftlimit)
        newbox = (bbox[0] - step, bbox[1], bbox[2], bbox[3])
        samples.append(newbox)
    #right
    if rightlimit > 1:
        step = np.random.randint(1, rightlimit)
        newbox = (bbox[0] + step, bbox[1], bbox[2], bbox[3])
        samples.append(newbox)
    # up
    if uplimit > 1:
        step = np.random.randint(1, uplimit)
        newbox = (bbox[0], bbox[1] + step, bbox[2], bbox[3])
        samples.append(newbox)
    # down
    if downlimit > 1:
        step = np.random.randint(1, downlimit)
        newbox = (bbox[0], bbox[1] - step, bbox[2], bbox[3])
        samples.append(newbox)

    return samples

def bboxScale(frame, bbox, max_scale=0.08, sample_size=3):
    height, width, channels = frame.shape
    max_h = int(max_scale * bbox[3])
    max_w = int(max_scale * bbox[2])
    if max_h <  2:
        max_h = 2
    if max_w < 2:
        max_w = 2
    uplimit = min(height - (bbox[1] + bbox[3]), max_h)
    downlimit = min(bbox[1], max_h)
    leftlimit = min(bbox[0], max_w)
    rightlimit = min(width - (bbox[0] + bbox[2]), max_w)
    max_width_scale = min(leftlimit, rightlimit)
    max_height_scale = min(uplimit, downlimit)
    samples = []

    if max_width_scale > 1 and max_height_scale > 1:
        for i in range(sample_size):
            step_w = np.random.randint(1, max_width_scale) * np.random.randint(-1, 2)
            step_h = np.random.randint(1, max_height_scale) * np.random.randint(-1, 2)
            newbox = (bbox[0] - step_w, bbox[1] - step_h, bbox[2] + 2 * step_w, bbox[3] + 2 * step_h)
            samples.append(newbox)

    return samples

def generate_positives(frame, bbox, max_shift=0.08, max_scale=0.08, nparray=True):
    bbox = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    positives = []
    # roi = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    positives.append(FeatureExtractor.extract_hog_hsv(frame, bbox))
    # shift in 4 directions
    shift_samples = bboxShift(frame, bbox, max_shift=max_shift)
    for newbox in shift_samples:
        positives.append(FeatureExtractor.extract_hog_hsv(frame, newbox))
    # flip horizontally
    roi = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    positives.append(FeatureExtractor.extract_hog_hsv(cv2.flip(roi, 1)))
    positives.append(FeatureExtractor.extract_hog_hsv(cv2.GaussianBlur(roi,(5,5),15)))
    # 3 scaled samples
    scale_samples = bboxScale(frame, bbox, max_scale=max_scale)
    for newbox in scale_samples:
        positives.append(FeatureExtractor.extract_hog_hsv(frame, newbox))
    if nparray:
        return np.array(positives)
    else:
        return positives

if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    #bbox = None
    bbox = cv2.selectROI(frame, False)
    print(bbox)
    print("positives: " + str(np.shape(generate_positives(frame, bbox))))
    '''shift = bboxShift(frame, bbox)
    for newbox in shift:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)'''
    '''scale = bboxScale(frame, bbox)
    for newbox in scale:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("image",frame)
    cv2.waitKey(0)'''
    roi = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    #roi = cv2.flip(roi, 1)
    roi = cv2.GaussianBlur(roi,(5,5),15)
    cv2.imshow("image",roi)
    cv2.waitKey(0)