import cv2
import numpy as np

hog = cv2.HOGDescriptor()
def extract_hog(frame, bbox=None):
    # extract HOG feature from bbox on frame
    ROI = frame
    if bbox is not None:
        ROI = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    height, width, channels = ROI.shape
    #print(height, width)
    #print(bbox[1], bbox[1]+bbox[3], bbox[0], bbox[0]+bbox[2])
    ROI = cv2.resize(ROI,(64,128),interpolation=cv2.INTER_CUBIC)
    h = hog.compute(ROI)
    h = h.reshape(3780)
    return h

def extract_color(frame, bbox=None):
    # extract color
    ROI = frame
    if bbox is not None:
        ROI = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    img_h = ROI[..., 0].flatten() # [0,179]
    img_s = ROI[..., 1].flatten() # [0,255]
    # v = frame[..., 2].flatten() * 100
    bin_width = 4
    h = np.zeros(45)
    s = np.zeros(64)
    for i in img_h:
        h[int(i / bin_width)] = h[int(i / bin_width)] + 1
    for i in img_s:
        s[int(i / bin_width)] = s[int(i / bin_width)] + 1
    h = h / len(img_h)
    s = s / len(img_s)
    return np.hstack((h,s))

def extract_hog_hsv(frame, bbox=None):
    f1 = extract_hog(frame, bbox)
    f2 = extract_color(frame, bbox)
    features = np.hstack((f1, f2))
    return features

if __name__ == "__main__":
    frame = cv2.imread("../figures/ewan.jpg")
    #bbox = None
    bbox = cv2.selectROI(frame, False)
    print("HOG descriptor shape: " + str(np.shape(extract_hog(frame, bbox=bbox))))
    print("HSV histogram descriptor shape: " + str(np.shape(extract_color(frame, bbox=bbox))))
    print("Feature shape: " + str(np.shape(extract_hog_hsv(frame, bbox=bbox))))
    