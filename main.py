import os, sys
import argparse
import kairos_face
import cv2
import numpy as np
import glob
import json
from termcolor import colored, cprint
import ConfigParser
import thread

config = ConfigParser.ConfigParser()
config.read('Config.ini')
appKey = config.items("API_KEY")
kairos_face.settings.app_id = appKey[0][1]
kairos_face.settings.app_key = appKey[1][1]
global DATASET_DIR
DATASET_DIR='/home/unicornx/Documents/DataSet/Face/'
Face_Class = ['Andy', 'River', 'Lennon']
TEMP_IMG = 'tmp.jpg'

def parseArg():

    parser = argparse.ArgumentParser(description='Face Recognition.')
    parser.add_argument('-i', '--IMG_URL', default='./1.jpg', help='Image URL Path')
    parser.add_argument('-d', '--DATA_DIR', default='./DataSet/', help='DataSet Image Dir')
    parser.add_argument('-w', '--WEBCAM', action='store_true', help='Enable WebCamera Mode')
    args = parser.parse_args()
    print(args)
    return args

def MultiEnroll(subjectID):
    ClassFolder = os.path.join(DATASET_DIR, subjectID)
    print(ClassFolder)
    try:
        _ = map(lambda imgPath: kairos_face.enroll_face(file=imgPath, subject_id=subjectID, gallery_name='a-gallery'), glob.glob(ClassFolder+'/*'))
    except:
        print('Exception Info : ', sys.exc_info())
def WebCamThread(string, sleeptime, *args):
    while(True):
        Recognition(FILE=TEMP_IMG)

def Recognition(URL=None, FILE=None):

    try:
        recognized_faces = kairos_face.recognize_face(url=URL, file=FILE, gallery_name='a-gallery')
        encodedjson =  json.dumps(recognized_faces)
        data = json.loads(encodedjson)
        data_transac = data['images'][0]['transaction']
        unknownFace = filter(lambda face: face['transaction']['status'] == 'failure', data['images'])
        knownFace = map(lambda face : face['transaction']['subject_id'],filter(lambda face: face['transaction']['status'] == 'success', data['images']))
	
	#print(knownFace)
	#print('unknown num : ', len(unknownFace))
	cprint('%s and %d unknown' % (knownFace, len(unknownFace)), 'yellow')
        #if data_transac['status'] == 'success':
        #    IsInClass = map(lambda c: data_transac['subject_id'] == c, Face_Class)
        #    reString = 'subject_id : %s, confidence : %s' % (data_transac['subject_id'], data_transac['confidence']) \
        #    if reduce(lambda c1,c2 : c1 or c2, IsInClass ) else 'unknow'
        #else:
        #    reString = 'unknow'
        #cprint(reString, 'yellow')
    except:
        #print ('Exception Info : ', sys.exc_info())
        cprint('Face not found.', 'red')

def main(args):
    _args = parseArg()
    DATASET_DIR=_args.DATA_DIR
    #global DATASET_DIR
    cprint('Enroll Training Set', 'green')
    _ = map(lambda x: MultiEnroll(x), Face_Class)
    
    if not _args.WEBCAM:
        Recognition(FILE=_args.IMG_URL)
        sys.exit()

    thread.start_new_thread(WebCamThread, ("ThreadFun", 1))
    cap = cv2.VideoCapture(0)
    while True:
        try:
            #nb = raw_input('Input a Image URL: ')
	    ret, frame = cap.read()
	    cv2.imshow('frame', frame)
            cv2.imwrite(TEMP_IMG, frame)
            #Recognition(FILE='./1.jpg')
            
	    if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("\n<<<<<<Exit>>>>>>>")
            sys.exit()

	
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
