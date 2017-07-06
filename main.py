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
Face_Class = ['Lennon', 'Jason']
TEMP_IMG = 'tmp.jpg'

def parseArg():

    parser = argparse.ArgumentParser(description='Face Recognition.')
    parser.add_argument('-i', '--IMG_FILE', default=None, help='Image URL Path')
    parser.add_argument('-u', '--IMG_URL', default=None, help='Image URL Path')
    parser.add_argument('-d', '--DATA_DIR', default='./DataSet/Private/', help='DataSet Image Dir')
    parser.add_argument('-w', '--WEBCAM', action='store_true', help='Enable WebCamera Mode')
    parser.add_argument('-c', '--CLEAN', action='store_true', help='Clean Gallery')
    args = parser.parse_args()
    cprint(args, 'green')
    return args

def MultiEnroll(subjectID):
    ClassFolder = os.path.join(DATASET_DIR, subjectID)
    print(ClassFolder)
    try:
        _ = map(lambda imgPath: kairos_face.enroll_face(file=imgPath, subject_id=subjectID, gallery_name='a-gallery'), glob.glob(ClassFolder+'/*'))
    except:
        print('Exception Info : ', sys.exc_info())
def WebCamThread():
    t = threading.currentThread()
    while getattr(t, 'do_run', True):
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
    
    if _args.CLEAN: 
        cprint('Galleries : {}' % (kairos_face.get_galleries_names_list()), 'green')
        _ = map(lambda galleryName: kairos_face.remove_gallery(galleryName) ,kairos_face.get_galleries_names_list()['gallery_ids'])
        if len(_) > 0 : print(_)
        sys.exit()
    
    global DATASET_DIR
    DATASET_DIR=_args.DATA_DIR
    cprint('Enroll Training Set', 'green')
    _ = map(lambda x: MultiEnroll(x), Face_Class)
    

    if not _args.WEBCAM:
        Recognition(URL=_args.IMG_URL, FILE=_args.IMG_FILE)
        sys.exit()

    #thread.start_new_thread(WebCamThread, ("ThreadFun", 1))
    thread1 = threading.Thread(target=WebCamThread)
    thread1.start()
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
            thread1.do_run=False
            thread1.join()
            sys.exit()

	
    # When everything done, release the capture
    thread1.do_run=False
    thread1.join()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
