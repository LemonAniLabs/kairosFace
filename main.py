import os, sys
import argparse
import kairos_face
import glob
import json
from termcolor import colored, cprint
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('Config.ini')
appKey = config.items("API_KEY")
kairos_face.settings.app_id = appKey[0][1]
kairos_face.settings.app_key = appKey[1][1]

DATASET_DIR='/home/unicornx/Documents/DataSet/Face/'
Face_Class = ['Jason', 'ST', 'Jobs']

def parseArg():

    parser = argparse.ArgumentParser(description='Face Recognition.')
    parser.add_argument('-i', '--ImgURL', default='./Test1.jpg', help='Image URL Path')
    args = parser.parse_args()
    return args

def MultiEnroll(subjectID):
    ClassFolder = os.path.join(DATASET_DIR, subjectID)
    print ClassFolder
    _ = map(lambda imgPath: kairos_face.enroll_face(file=imgPath, subject_id=subjectID, gallery_name='a-gallery'), glob.glob(ClassFolder+'/*'))
    #kairos_face.enroll_face(file=imgPath, subject_id=subjectID, gallery_name='a-gallery')

def Recognition(URL):

    try:
        recognized_faces = kairos_face.recognize_face(url=URL, gallery_name='a-gallery')
        encodedjson =  json.dumps(recognized_faces)
        data = json.loads(encodedjson)
        data_transac = data['images'][0]['transaction']
        if data_transac['status'] == 'success':
            IsInClass = map(lambda c: data_transac['subject_id'] == c, Face_Class)
            reString = 'subject_id : %s, confidence : %s' % (data_transac['subject_id'], data_transac['confidence']) \
            if reduce(lambda c1,c2 : c1 or c2, IsInClass ) else 'unknow'
        else:
            reString = 'unknow'
        cprint(reString, 'yellow')
    except:
        print ('Exception Info : ', sys.exc_info()[0])
        cprint('Face not find.', 'red')

def main(args):
    _args = parseArg()
    
    cprint('Enroll Training Set', 'green')
    _ = map(lambda x: MultiEnroll(x, ), Face_Class)

    while True:
        try:
            nb = raw_input('Input a Image URL: ')
            Recognition(nb)

        except KeyboardInterrupt:
            print "\n<<<<<<Exit>>>>>>>"
            sys.exit()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
