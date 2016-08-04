import openface
import cv2
import os
import pickle
import pdb
import dlib

##
# Face detection on small image, face identification on large
#
##

import numpy as np
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

facePredictorFile = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
classifierFile = os.path.join(fileDir, 'generated', 'classifier.pkl')
torchNetworkModelFile = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_profileface.xml')


class Face:
    def __init__(self, box, rep):
        self.box = box
        self.rep = rep

class Person:
    def __init__(self, name, face, confidence):
        self.name = name
        self.face = face
        self.confidence = confidence


# from http://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap
def squares_intersect(s1, s2):
    s1x1 = s1.left()
    s1y1 = s1.top()
    s1x2 = s1x1 + s1.height()
    s1y2 = s1y1 + s1.width()
    s2x1 = s2.left()
    s2y1 = s2.top()
    s2x2 = s2x1 + s2.height()
    s2y2 = s2y1 + s2.width()

    si = max(0, max(s1x2, s2x2) - min(s1x1, s2x1)) * max(0, max(s1y2, s2y2) - min(s1y1, s2y1))
    su = s1.width() * s1.height() + s2.width() * s2.height() - si

    print("%s, %s, %s, %s - %s, %s, %s, %s - %s" %(s1x1, s1y1, s1x2, s1y2, s2x1, s2y1, s2x2, s2y2, si/su))
    return si / su

def is_false_positive(img, box):
    box = cv2_rect_to_dlib(box)
    shape = np.shape(img)
    x1 = box.left()
    y1 = box.top()
    x2 = min(x1 + box.width(), shape[1])
    y2 = min(y1 + box.height(), shape[0])
    #pdb.set_trace()
    faces = align.getAllFaceBoundingBoxes(img)

    print(faces)
    return len(faces) == 0


def cv2_rect_to_dlib(rect):
    x1 = long(rect[0])
    y1 = long(rect[1])
    x2 = long(rect[2]) + x1
    y2 = long(rect[3]) + y1
    
    return dlib.rectangle(x1, y1, x2, y2)

def get_faces_bounding_boxes_cv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_boxes = face_cascade.detectMultiScale(
        gray, 
        1.2, 
        3, 
        cv2.cv.CV_HAAR_DO_CANNY_PRUNING, 
        minSize=(50,50)
    )

    boxes = []

    for box in face_boxes:
        if True: #not is_false_positive(img, box):
            boxes.append(cv2_rect_to_dlib(box))
            print("Found face %s" % (box))
        else:
            print("Found false positive %s" % (box))

    return boxes

def get_faces_bounding_boxes_dlib(img):
    # Convert from dlib.rectangles to list
    return [x for x in align.getAllFaceBoundingBoxes(img)]


def get_tracked_person(box):
    for person in tracked_persons:
        if squares_intersect(person.face.box, box) > 0.5:
            return person

    return None
 
def getFaces(boxes, img):

    faces = []

    for box in boxes:
        tracked_person = get_tracked_person(box)
        if not tracked_person:
            aligned_face = align.align(96, img, box, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            rep = net.forward(aligned_face)
            faces.append(Face(box, rep))
        else:
            tracked_person.face.box = box

    return faces

def findPersons(img):
    with open(classifierFile, 'r') as f:
        (labels, classifier) = pickle.load(f)

    persons = []
    confidences = []
    
    for i, face in enumerate(faces):
        try:
            rep = face.rep.reshape(1, -1)
        except:
            #print "No Face detected"
            return (None, None, None)

        predictions = classifier.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        
        name = labels.inverse_transform(maxI)
        confidence = predictions[maxI]
        
        person = Person(name, face, confidence)
        persons.append(person)
        
    return persons


def draw_person_box(img, person):
    print("Drawing box for %s - confidence %.2f" % (person.name, person.confidence))

    box = person.face.box
    x1 = int(box.left())
    y1 = int(box.top())
    x2 = int(box.right())
    y2 = int(box.bottom())
    cv2.rectangle(img, (x1, y1), (x2, y2),(0, 255, 0),2)
    
    ts = cv2.getTextSize(person.name, cv2.FONT_HERSHEY_SIMPLEX, 2, 255)[0]
    cv2.putText(img, person.name, ((x2-x1)/2 + x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)


def prune_match_boxes_persons(boxes, persons):
    pruned_boxes = boxes[:]
    pruned_persons = []
    #pdb.set_trace()
    for person in persons:
        for i, box in enumerate(boxes):
            a = squares_intersect(person.face.box, box)
            #print("Match %s %s %.2f" % (person.face.box, box, a))
            if a > face_confidence_threshold:
                #print("Match %s %s" % (person.face.box, box))
                person.face.box = box
                pruned_persons.append(person)
                pruned_boxes[i] = None

    pruned_boxes = [b for b in pruned_boxes if b is not None]

    #print("Pruned boxes %s, Pruned persons %s" % (pruned_boxes, pruned_persons))
    return (pruned_boxes, pruned_persons)


def get_bounding_boxes(img):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    boxes = []
    
    #print("Checking for faces")
    boxes = face_detector(img)
    iteration = 0       

    return boxes

if __name__ == '__main__':
    face_detector = get_faces_bounding_boxes_dlib
    face_confidence_threshold = 0.5

    tracked_persons = []
    align = openface.AlignDlib(facePredictorFile)
    net = openface.TorchNeuralNet(torchNetworkModelFile, imgDim=96)
    iteration = 0
     
    vc = cv2.VideoCapture(0)
    # set(3, width) set(4, height)

    while True:
        _, frame = vc.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        #pdb.set_trace()
        if iteration % 3 == 0:
            boxes = get_bounding_boxes(frame)
            #pdb.set_trace()
            boxes, pruned_tracked_persons = prune_match_boxes_persons(boxes, tracked_persons)
 
            #pdb.set_trace()
            faces = getFaces(boxes, frame)
            tracked_persons = findPersons(faces) + pruned_tracked_persons

        for person in tracked_persons:
            draw_person_box(frame, person)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        iteration += 1

    vc.release()
    cv2.destroyAllWindows()
