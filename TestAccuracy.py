import random
import base64
import os, sys
import argparse
import kairos_face
import cv2
import numpy as np
import glob
import json
from termcolor import colored, cprint
import ConfigParser
import threading

config = ConfigParser.ConfigParser()
config.read('Config.ini')
appKey = config.items("API_KEY")
kairos_face.settings.app_id = appKey[0][1]
kairos_face.settings.app_key = appKey[1][1]
DATASET_DIR='/home/unicornx/Documents/DataSet/Face/'
Face_Class = ['Andy', 'River', 'Lennon']
TEMP_IMG = 'tmp.jpg'

def parseArg():

    parser = argparse.ArgumentParser(description='Face Recognition.')
    parser.add_argument('-i', '--IMG_URL', default='./1.jpg', help='Image URL Path')
    parser.add_argument('-d', '--DATA_DIR', default='./DataSet/Legislators', help='DataSet Image Dir')
    parser.add_argument('-w', '--WEBCAM', action='store_true', help='Enable WebCamera Mode')
    parser.add_argument('-c', '--CLEAN', action='store_true', help='Clean Gallery')
    args = parser.parse_args()
    cprint(args, 'green')
    return args

def MultiEnroll(subjectID, fileSet=None ,multiple=False):
    cprint('Enroll subject : %s '%(subjectID), 'yellow') 
    if fileSet is None:
        ClassFolder = os.path.join(DATASET_DIR, subjectID)
        print(ClassFolder)
        try:
            _ = map(lambda imgPath: kairos_face.enroll_face(file=imgPath, subject_id=subjectID,multiple_faces = multiple, gallery_name='b-gallery'), glob.glob(ClassFolder+'/*'))
        except:
            print('Exception Info : ', sys.exc_info())
    else:
        print fileSet
        try:
            _ = map(lambda aFile : kairos_face.enroll_face(file=aFile, subject_id=subjectID,multiple_faces = multiple, gallery_name='b-gallery'), fileSet)
        except:
            print('Exception Info : ', sys.exc_info())

def Recognition(subject_id, FILESET=None):

    try:
        cprint('Test id : %s ' %(subject_id), 'yellow')
        print(FILESET)
	FILESET = np.random.choice(FILESET,5)
        recognized_faces = map(lambda aFace : json.loads(json.dumps(kairos_face.recognize_face(file=aFace, gallery_name='b-gallery')))['images'][0]['transaction'], FILESET)
        #result = map(lambda aFaceInfo : aFaceInfo['status'] == 'success' and aFaceInfo['subject_id'] == subject_id ,recognized_faces)

        result = map(lambda aFaceInfo : aFaceInfo['status'] ,recognized_faces)
        
	#recognized_faces = kairos_face.recognize_face(url=URL, file=FILE, gallery_name='b-gallery')
        #encodedjson =  json.dumps(recognized_faces)
        #data = json.loads(encodedjson)
        #data_transac = data['images'][0]['transaction']
        #unknownFace = filter(lambda face: face['transaction']['status'] == 'failure', data['images'])
        #knownFace = map(lambda face : face['transaction']['subject_id'],filter(lambda face: face['transaction']['status'] == 'success', data['images']))
	print(result)
    except:
        print ('Exception Info : ', sys.exc_info())
        cprint('Face not found.', 'red')

def main(args):
    _args = parseArg()
    
    if _args.CLEAN: 
        cprint('Galleries : %s' % (kairos_face.get_galleries_names_list()), 'green')
        _ = map(lambda galleryName: kairos_face.remove_gallery(galleryName) ,kairos_face.get_galleries_names_list()['gallery_ids'])
        if len(_) > 0 : print(_)
        sys.exit()
    
    global DATASET_DIR
    DATASET_DIR=_args.DATA_DIR
    cprint('Enroll Training Set', 'green')

    random_class = random.sample(xrange(112), 3)
    print random_class
    TestClassFiles = map(lambda aClass : {'id' : aClass, 'data' : glob.glob(os.path.join(DATASET_DIR, str(aClass),'*'))}, random_class)
    splitedSet = map(lambda aClass : (aClass['id'], np.split(aClass['data'],[7,])), TestClassFiles)
    trainingSet = [(aSet[0], aSet[1][0]) for aSet in splitedSet]
    testingSet = [(aSet[0], aSet[1][1]) for aSet in splitedSet]
    
    ## Enroll Multiple Training Faces
    #print trainingSet[0]
    _ = map(lambda x: MultiEnroll(x[0],fileSet=x[1], multiple=True), trainingSet)
    

    _ = map(lambda x: Recognition(x[0], x[1]), testingSet)
    #Recognition(FILE=_args.IMG_URL)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
