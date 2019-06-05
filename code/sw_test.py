import slidingwindow as sw
import numpy as np
import cv2
import FeatureExtractor

frame = cv2.imread("ts.jpg")

windows = sw.generate(frame, sw.DimOrder.HeightWidthChannel, 256, 0.5)

#print(windows)
count = 0
print(len(windows))
for window in windows:
    f = window.apply(frame)
    #print(np.shape(f))
    #print(np.shape(f))
    #print(window)
    print(count)
    print(np.shape(f))
    count = count + 1

'''
for window in windows:
    bbox = window.getRect()
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)

while True:
    cv2.imshow("image",frame)
    cv2.waitKey(1)'''