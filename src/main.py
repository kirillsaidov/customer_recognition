# module main

# system
import os
import uuid
from datetime import datetime, timedelta

# db
import pymongo

# computer vision
import cv2

# face recognition
from deepface import DeepFace

# custom
from auxiliary import *

# constants
DEEPFACE_MODELS = [
    'VGG-Face',
    'Facenet',
    'Facenet512',
    'OpenFace',
    'DeepFace',
    'DeepID',
    'ArcFace',
    'Dlib',
    'SFace',
    'GhostFaceNet',
]

# db
mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
mongo_col = mongo_client['cafe_recognition']['log']


def process_face(frame: cv2.Mat, db_path: str, model: str, repeate_log_s: int = 60):
    # find face and analyze
    faces = DeepFace.find(img_path=frame, db_path=db_path, model_name=model, enforce_detection=False)[0]
    face_info = DeepFace.analyze(
        img_path = './imgs/img_id1.jpg', 
        actions = ['age', 'gender', 'race', 'emotion'], 
        enforce_detection=False,
    )[0]

    # log information
    if not faces.empty:
        # get face id
        face_id = os.path.basename(faces.iloc[0]['identity'])

        # find face id and log to db if 1 minute passed since last log (of that face id)
        entry = mongo_col.find_one({'id': face_id}, sort=[( '_id', pymongo.DESCENDING )])
        if entry and datetime.now() - datetime.strptime(entry['time'], '%d-%m-%Y %H:%M:%S') > timedelta(seconds=repeate_log_s):
            mongo_col.insert_one({
                'time': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                'status': 'found',
                'id': face_id,
                'age': face_info['age'],
                'gender': face_info['dominant_gender'],
                'race': face_info['dominant_race'],
                'emotion': face_info['emotion'],
            })
    else:
        # generate face id
        face_id = uuid.uuid4().hex + '.jpg'

        # save face
        cv2.imwrite(os.path.join(db_path, face_id), frame)

        # log to db
        mongo_col.insert_one({
            'time': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
            'status': 'new',
            'id': face_id,
            'age': face_info['age'],
            'gender': face_info['dominant_gender'],
            'race': face_info['dominant_race'],
            'emotion': face_info['emotion'],
        })


def update(params: dict):
    frame = params['frame']
        
    # process events
    key = cv2.waitKey(30)
    if (key & 0xFF == ord('h')) or key == 40:
        params['enable_debug_info'] = not params['enable_debug_info']
    if (key & 0xFF == ord('r')) or key == 40:
        params['enable_recognition'] = not params['enable_recognition']
        params['show_tip'] = False
        params['face_bbox'] = None
   
    # detect faces
    if params['enable_recognition']:
        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
        face = faces[0]['facial_area']
        if face['left_eye'] or face['right_eye']:
            params['face_bbox'] = [
                (face['x'], face['y']),
                (face['x'] + face['w'], face['y'] + face['h']),
            ]
        else:
            params['face_bbox'] = None

        # process face
        if params['face_bbox']:
            fb = params['face_bbox'] 
            fr = params['face_region']
            face_bbox_area = (fb[1][0] - fb[0][0]) * (fb[1][1] - fb[0][1]) 
            face_region_area = ((fr[0] * frame.shape[1]) * (fr[1] * frame.shape[0]))
            ratio = face_bbox_area / face_region_area
            if ratio > 1.5:
                frame_crop = frame[fb[0][1]:fb[1][1], fb[0][0]:fb[1][0]]

                # show tip
                params['show_tip'] = False 

                # process
                process_face(frame_crop, db_path=params['db_path'], model=params['deepface_model'], repeate_log_s=params['repeat_log_s'])
            else:
                params['show_tip'] = True
    
    # draw                    
    draw(frame, params)
    
    return frame


def draw(frame: cv2.Mat, params: dict = None):
    # define data
    data = [
        f'>> "h" to hide debug information ({"on" if params["enable_debug_info"] else "off"})',
        f'>> "r" to enable/disable recognition ({"on" if params["enable_recognition"] else "off"})',
        f'>> "q" to quit',
        f'>> CLOSER, LOOK AT CAMERA',
    ]
    
    # draw data
    if params['enable_debug_info']:
        for i, entry in enumerate(data):
            if i == len(data) - 1 and params['show_tip']:
               cv2.putText(
                    img=frame,
                    text=entry,
                    org=(15, 30*(i+1)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.9,
                    color=rgb2bgr(Colors.MAGENTA),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            elif i < len(data) - 1:
                cv2.putText(
                    img=frame,
                    text=entry,
                    org=(15, 30*(i+1)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.9,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
    
    # draw face box
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    p1 = (
        int((frame_width - frame_width*params['face_region'][0])/2),
        int((frame_height - frame_width*params['face_region'][1])/2),
    )
    p2 = (
        int((frame_width + frame_width*params['face_region'][0])/2),
        int((frame_height + frame_width*params['face_region'][1])/2),
    )
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=rgb2bgr(Colors.LIME), thickness=4)

    # draw face bbox
    if params['face_bbox']:
        cv2.rectangle(frame, pt1=params['face_bbox'][0], pt2=params['face_bbox'][1], color=rgb2bgr(Colors.CORAL), thickness=3)


if __name__ == '__main__':
    update_params = {
        'enable_debug_info': True,
        'enable_recognition': False,
        'face_region': (0.3, 0.4),
        'face_bbox': None,
        'db_path': './imgs',
        'deepface_model': DEEPFACE_MODELS[1],
        'repeat_log_s': 60,
        'show_tip': False, 
    }
        
    # run
    display_video(
        video_path=0, 
        func=update, 
        func_params=update_params, 
        frame_rate=12,
        draw_fps=True, 
        frame_size=(720, 480),
        wait_per_frame_secs=0.5,
        record_file_name=None,
    )


