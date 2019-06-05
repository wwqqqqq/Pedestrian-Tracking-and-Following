import cv2
import numpy as np
#import Detector
import classifier
import imutils
#import extract


def extract(path):
    img_path = path + '/img/'
    rect = path + '/groundtruth_rect.txt'
    frames = []
    boxes = []
    count = 0
    with open(rect,'r') as f:
        content = f.read()
        lines = content.split('\n')
        for line in lines:
            count = count + 1
            try:
                if ',' in line:
                    box = eval(line)
                else:
                    nums = line.split('\t')
                    box = (int(nums[0]),int(nums[1]),int(nums[2]),int(nums[3]))
                #print(box)
                boxes.append(box)
                ind = str(count)
                if len(ind) < 4:
                    ind = "0" * (4-len(ind)) + ind
                frames.append(cv2.imread(img_path+ind+".jpg"))
            except:
                break
    return frames, boxes

def outofRange(frame, bbox, threshold=10):
    #if bbox[0] >= 0 and bbox[1] >= 0 and bbox[0] + bbox[2] <= frame.shape[1] and bbox[1] + bbox[3] <= frame.shape[0]:
    #    return bbox
    newbox = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
    if newbox[0] < 0:
        newbox[2] = newbox[2] + newbox[0]
        newbox[0] = 0
    if newbox[1] < 0:
        newbox[3] = newbox[3] + newbox[1]
        newbox[1] = 0
    if newbox[0] + newbox[2] > frame.shape[1]:
        newbox[2] = frame.shape[1] - newbox[0]
    if newbox[1] + newbox[3] > frame.shape[0]:
        newbox[3] = frame.shape[0] - newbox[1]
    return (newbox[0],newbox[1],newbox[2],newbox[3])



def track_sequence(ctype='none', start_classifier=100, record=False, max_failure=5, fail_thred=0.7, success_thred=0.7, record_path="test.avi", dataset="Basketball", neg_sample=10, scale=1.2, result_path="test.txt"):

    # Set up tracker.
    tracker = cv2.TrackerCSRT_create()
    tracker2 = cv2.TrackerCSRT_create()
    #tracker = cv2.TrackerKCF_create()

    frames, boxes = extract(dataset)
    print(len(frames), len(boxes))

    #bbox = cv2.selectROI(frame, False)

    #bbox = (125, 18, 127, 258)
    bbox = boxes[0]
    frame = frames[0]
    print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    tracker2.init(frame,bbox)
    sum_freq = 0
    count = 0

    init_height = bbox[3]
    init_width = bbox[2]
    
    # write the result to video
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter(record_path, fourcc, 7, (frame.shape[1],frame.shape[0]))

    print("Initialization done!")
    train_frames = [frame]
    train_bbox = [bbox]
    trained = False

    count_clf_err = 0

    f = open("results/"+result_path, 'w')
    

    while True:
        count = count + 1
        try:
            frame = frames[count]
            rect = boxes[count]
        except:
            break
        # Read a new frame
        #frame = imutils.resize(frame, width=min(frame.shape[1], 500))
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
        ok2,bbox2 = tracker2.update(frame)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        sum_freq = sum_freq + fps

        # Draw bounding box
        if ok and bbox != (0,0,0,0):# and not outofRange(frame, bbox):
            if not trained:
                try:
                    newbox = outofRange(frame, bbox)
                    train_frames.append(frame)
                    train_bbox.append(bbox)
                    if count == start_classifier:
                        ## 防止训练时数据量过大，造成严重延迟
                        if ctype == 'adaboost':
                            classifier.fit_adaboost(train_frames,train_bbox, plural=True)
                        elif ctype == 'svm':
                            classifier.fit_svc(train_frames, train_bbox, plural=True)
                        print("Classifier Trained!")
                        trained = True
                except:
                    print("Something is wrong with fitting.")
            p1 = (int(bbox2[0]), int(bbox2[1]))
            p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
            
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            
            if trained and count % 10 == 1 and ctype != 'none':
                # verify the tracker
                newbox = outofRange(frame, bbox)
                try:
                    if ctype == 'svm':
                        pred = classifier.positive_score(frame, newbox)
                        print(pred)
                        print("GroundTruth pred:", classifier.positive_score(frame, rect))
                    else:
                        pred =  classifier.predict(frame,newbox,classifier=ctype)
                    if pred < fail_thred:
                        print("tracking may fail!!")
                        count_clf_err = count_clf_err + 1
                        if count_clf_err > max_failure:
                            ok = False
                            count_clf_err = 0
                            print("failure due to exceed max detector err!")
                    else:
                        count_clf_err = 0
                except:
                    print("Something wrong with this bbox...",newbox)
                    #ok = False
        if not ok or bbox == (0,0,0,0):# or outofRange(frame, bbox):
            #print(bbox)
            # Tracking failure
            # try recover
            #rec = recognize(frame)
            print("FAILURE!!")
            # train classifier
            if not trained:
                print("train classifier")
                #classifier.fit_svc(train_frames, train_bbox, plural=True)
                if ctype == 'adaboost':
                    classifier.fit_adaboost(train_frames,train_bbox, plural=True)
                elif ctype == 'svm':
                    classifier.fit_svc(train_frames, train_bbox, plural=True)
                print("Classifier Trained!")
                trained = True
            if ctype != 'none':
                score, newbox = classifier.SlidingWindow(frame, overlap=0.8, windowSize=[init_width,init_height], classifier=ctype, scale=scale)
            else:
                score = 0
            print("new score: ",score)
            if score > success_thred and newbox != (0,0,0,0):
                bbox = newbox
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)
                #init_height = bbox[3]
                #init_width = bbox[2]
                if not ok:
                    print("Tracker init failed.")
                    break
                print("Initialize Tracker again... Score = ",score)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                print("SVM!!!")
            else:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                #cv2.putText(frame, "Tracking failure detected", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        p1 = (int(rect[0]), int(rect[1]))
        p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
        #cv2.putText(frame, "Blue: CSR-DCF with SVM", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);
        '''if ok2:
            cv2.putText(frame, "Green: CSR-DCF only", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2);
        else:
            cv2.putText(frame, "Green: CSR-DCF only failed!", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2);'''
        #cv2.putText(frame, "Red: Groundtruth", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);

        f.write(str(rect)+str(bbox)+str(bbox2)+"\r\n")
     
        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)) + " AVG : " + str(int(sum_freq / count)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("tracking", frame)
        if record:
            outVideo.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1)
    if record:
        print("finish recording.")
    f.close()





track_sequence(ctype='none', dataset="Girl2",fail_thred=0.7, success_thred=0.5, max_failure=2, record=False, record_path="Girl2.avi", result_path="Girl2.txt", start_classifier=100, neg_sample=20, scale=1.5)





def fps_sum(ctype='none', start_classifier=100, record=False, max_failure=5, fail_thred=0.7, success_thred=0.7, record_path="test.avi", dataset="Basketball", neg_sample=10, scale=1.2, result_path="test.txt"):

    # Set up tracker.
    tracker = cv2.TrackerCSRT_create()
    #tracker = cv2.TrackerKCF_create()

    frames, boxes = extract(dataset)

    #bbox = cv2.selectROI(frame, False)

    #bbox = (125, 18, 127, 258)
    bbox = boxes[0]
    frame = frames[0]
    print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    sum_freq = 0
    count = 0

    init_height = bbox[3]
    init_width = bbox[2]

    print("Initialization done!")
    train_frames = [frame]
    train_bbox = [bbox]
    trained = False

    count_clf_err = 0
    count_frame = 0
    if ctype=='none':
        trained = True
        timer = cv2.getTickCount()
        count_frame = 0
    

    while True:
        count = count + 1
        count_frame = count_frame + 1
        try:
            frame = frames[count]
            rect = boxes[count]
        except:
            break
        # Read a new frame
        #frame = imutils.resize(frame, width=min(frame.shape[1], 500))
        
 
        # Update tracker
        ok, bbox = tracker.update(frame)        
        # Calculate Frames per second (FPS)


        # Draw bounding box
        if ok and bbox != (0,0,0,0):# and not outofRange(frame, bbox):
            if not trained:
                try:
                    newbox = outofRange(frame, bbox)
                    train_frames.append(frame)
                    train_bbox.append(bbox)
                except:
                    print("Something is wrong with bbox.")
                if count == start_classifier:
                    if ctype == 'adaboost':
                        classifier.fit_adaboost(train_frames,train_bbox, plural=True)
                    elif ctype == 'svm':
                        classifier.fit_svc(train_frames, train_bbox, plural=True)
                    print("Classifier Trained!")
                    trained = True
                    timer = cv2.getTickCount()
                    count_frame = 0
                
            if trained and count % 10 == 1 and ctype != 'none':
                # verify the tracker
                newbox = outofRange(frame, bbox)
                try:
                    if ctype == 'svm':
                        pred = classifier.positive_score(frame, newbox)
                    else:
                        pred =  classifier.predict(frame,newbox,classifier=ctype)
                    if pred < fail_thred:
                        count_clf_err = count_clf_err + 1
                        if count_clf_err > max_failure:
                            ok = False
                            count_clf_err = 0
                    else:
                        count_clf_err = 0
                except:
                    print("Something wrong with this bbox...",newbox)
                    #ok = False
        if not ok or bbox == (0,0,0,0):# or outofRange(frame, bbox):
            if not trained:
                classifier.fit_svc(train_frames, train_bbox, plural=True)
                trained = True
                print("begin")
                timer = cv2.getTickCount()
                count_frame = 0
            if ctype != 'none':
                score, newbox = classifier.SlidingWindow(frame, overlap=0.8, windowSize=[init_width,init_height], classifier=ctype, scale=scale)
            else:
                score = 0
            if score > success_thred and newbox != (0,0,0,0):
                bbox = newbox
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)
                #init_height = bbox[3]
                #init_width = bbox[2]
                if not ok:
                    print("Tracker init failed.")
                    break
    print((cv2.getTickCount() - timer)/cv2.getTickFrequency())
    print(count_frame)


#fps_sum(ctype='none', dataset="Walking2",fail_thred=0.7, success_thred=0.5, max_failure=2, record=True, record_path="Girl2.avi", result_path="Girl2.txt", start_classifier=100, neg_sample=20, scale=1.5)

#fps_sum(ctype='svm', dataset="Walking2",fail_thred=0.7, success_thred=0.5, max_failure=2, record=True, record_path="Girl2.avi", result_path="Girl2.txt", start_classifier=100, neg_sample=20, scale=1.5)

'''
Basketball
BlurBody
Gym
Human2
Human7
Jogging_1
Jogging_2
Singer1
Skater
Skater2
Skating2
Walking2
Woman
Human8
'''









#if __name__ == '__main__' :
if False:
    ctype = "none"
 
    # Set up tracker.
    tracker = cv2.TrackerCSRT_create()
    tracker2 = cv2.TrackerCSRT_create()
    #tracker = cv2.TrackerKCF_create()
 
    # Read video
    #video = cv2.VideoCapture(0)
    #video = cv2.VideoCapture("ticao20190514_170124.mp4")
    #video = cv2.VideoCapture("test_video.mp4")
    #video = cv2.VideoCapture("dancing20190516_205255.mp4")
    #video = cv2.VideoCapture("dancing20190516_215934.mp4")
 
    # Exit if video not opened.
    #if not video.isOpened():
    #    print("Could not open video")
    #    sys.exit()
 
    # Read first frame.
    #ok, frame = video.read()
    count = 308
    ok = True
    #frame = cv2.imread("pedxing-seq1/0000"+str(1600+count)+".jpg")
    frames, boxes = extract("Basketball")
    if not ok:
        print('Cannot read video file')
        sys.exit()
    #frame = imutils.resize(frame, width=min(frame.shape[1], 500))
    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)
    #bbox = cv2.selectROI(frame, False)
    i = (186, 231, 140, 131)#, (574, 310, 521, 217), (571, 177, 615, 68), (567, 310, 628, 403)
    bbox = (min(i[0],i[2]),min(i[1],i[3]),abs(i[0]-i[2]),abs(i[1]-i[3]))
    #bbox = (125, 18, 127, 258)
    print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    tracker2.init(frame,bbox)
    #classifier.fit_svc(frame, bbox)
    #classifier.fit_adaboost(frame,bbox)
    sum_freq = 0

    init_height = bbox[3]
    init_width = bbox[2]
    
    # write the result to video
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter('svm.avi', fourcc, 7, (frame.shape[1],frame.shape[0]))

    print("Initialization done!")
    train_frames = [frame]
    train_bbox = [bbox]
    trained = False

    count_clf_err = 0
    
    baseline = extract.extract(1)
    base = baseline["pedxing-seq1/0000"+str(1600+count)+".jpg"]

    while True:
        count = count + 1
        # Read a new frame
        #ok, frame = video.read()
        ok = True
        filename = "pedxing-seq1/0000"+str(1600+count)+".jpg"
        frame = cv2.imread(filename)
        if not ok:
            break
        #frame = imutils.resize(frame, width=min(frame.shape[1], 500))

        #if count < 100:
        '''elif count == 100:
            # train classifier
            print("train classifier")
            classifier.fit_svc(train_frames, train_bbox, plural=True)
            #classifier.fit_adaboost(train_frames,train_bbox, plural=True)
            print("Classifier Trained!")'''
        
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
        ok2,bbox2 = tracker2.update(frame)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        sum_freq = sum_freq + fps

        try:
            temp = baseline[filename]
            base = temp
            s = ''
            for i in base:
                x = (i[0] + i[2]) / 2
                y = (i[1] + i[3]) / 2
                s = s + str((x,y))
            print("Base: "+s+" Bbox: "+str((bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2)))
        except:
            pass

        # Draw bounding box
        if ok and bbox != (0,0,0,0):# and not outofRange(frame, bbox):
            if not trained:
                newbox = outofRange(frame, bbox)
                train_frames.append(frame)
                train_bbox.append(bbox)
                if count == 100:
                    ## 防止训练时数据量过大，造成严重延迟
                    if ctype == 'adaboost':
                        classifier.fit_adaboost(train_frames,train_bbox, plural=True)
                    elif ctype == 'svm':
                        classifier.fit_svc(train_frames, train_bbox, plural=True)
                    print("Classifier Trained!")
                    trained = True
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            p1 = (int(bbox2[0]), int(bbox2[1]))
            p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
            if trained and count % 10 == 1 and ctype != 'none':
                # verify the tracker
                newbox = outofRange(frame, bbox)
                try:
                    if ctype == 'svm':
                        pred = classifier.positive_score(frame, newbox)
                        print(pred)
                    else:
                        pred =  classifier.predict(frame,newbox,classifier=ctype)
                    if pred < 0.7:
                        print(bbox, "tracking may fail!!")
                        count_clf_err = count_clf_err + 1
                        if count_clf_err > 2:
                            ok = False
                    else:
                        count_clf_err = 0
                except:
                    print("Something wrong with this bbox...",newbox)
                    #ok = False
        if not ok or bbox == (0,0,0,0):# or outofRange(frame, bbox):
            #print(bbox)
            # Tracking failure
            # try recover
            #rec = recognize(frame)
            print("FAILURE!!")
            # train classifier
            if not trained:
                print("train classifier")
                #classifier.fit_svc(train_frames, train_bbox, plural=True)
                if ctype == 'adaboost':
                    classifier.fit_adaboost(train_frames,train_bbox, plural=True)
                elif ctype == 'svm':
                    classifier.fit_svc(train_frames, train_bbox, plural=True)
                print("Classifier Trained!")
                trained = True
            if ctype != 'none':
                score, newbox = classifier.SlidingWindow(frame, stepSize=min(30,int(0.3*init_width), int(0.3*init_height)), windowSize=[init_width,init_height], classifier=ctype)
            else:
                score = 0
            if score > 0.7 and newbox != (0,0,0,0):
                bbox = newbox
                tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)
                if not ok:
                    print("Tracker init failed.")
                    break
                print("Initialize Tracker... Again...")
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                print("SVM!!!")
            else:
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        for i in base:
            p1 = (int(i[0]), int(i[1]))
            p2 = (int(i[2]), int(i[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
        cv2.putText(frame, "Blue: CSR-DCF with SVM", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        cv2.putText(frame, "Green: CSR-DCF only" + str(int(sum_freq / count)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
     
        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)) + " AVG : " + str(int(sum_freq / count)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("tracking", frame)

        outVideo.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1)
    print("finish recording!")
    #classifier.fit_svc(train_frames, train_bbox, plural=True)
    #classifier.fit_adaboost(train_frames,train_bbox, plural=True)