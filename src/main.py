# module main

# system
import os
import uuid
import json
import traceback
from queue import Queue
from threading import Thread, Event
from datetime import datetime, timedelta

# db
import pymongo

# computer vision
import cv2

# face recognition
from deepface import DeepFace

# custom
from db import *
from cvutil import *

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
mongo_col = mongo_client['cf']['log']
export_csv = 'db.csv'


def server_face_processing(fq: Queue, shutdown: Event, db_path: str, model_name: str, repeat_log_s: int, timeout: float = 0.25):
    """Server for processing faces and logging to DB

    Args:
        fq (Queue): queue
        shutdown (Event): shutdown event
        timeout (float, optional): timeout for receiving data from queue. Defaults to 0.25.
    """
    def save_new_face(frame: cv2.Mat):
        # generate id
        face_id = uuid.uuid4().hex + '.jpg'

        # save face image
        cv2.imwrite(os.path.join(db_path, face_id), frame)

        # age and gender
        face_info = DeepFace.analyze(
            img_path = frame, 
            actions = ['age', 'gender'], 
            enforce_detection=False,
            silent=True,
        )[0]

        # log
        db_log_insert(col=mongo_col, face_id=face_id, status='new', info={
            'age': face_info['age'],
            'gender': face_info['dominant_gender'],
        }, export_csv=export_csv)
 
    # start
    try:
        # check db exists
        if not os.path.exists(db_path):
            os.mkdir(db_path)

        # run processing server
        while not shutdown.is_set():
            # get frame
            frame = None
            try: frame = fq.get(timeout=timeout)
            except: continue

            # process frame
            dir_list = list(filter(lambda x: x.endswith('.jpg'), os.listdir(db_path)))
            if len(dir_list):
                faces = DeepFace.find(img_path=frame, db_path=db_path, model_name=model_name, enforce_detection=False, silent=True)
                if len(faces) and not faces[0].empty:                    
                    # find face id and log to db if N seconds have passed since last log (of that face id)
                    face = faces[0].to_dict()
                    print(face)
                    face_id = os.path.basename(face['identity'][0])
                    entry = mongo_col.find_one({'id': face_id}, sort=[( '_id', pymongo.DESCENDING )])
                    if entry and datetime.now() - datetime.strptime(entry['time'], '%d-%m-%Y %H:%M:%S') > timedelta(seconds=repeat_log_s):
                        db_log_insert(col=mongo_col, face_id=face_id, status='found', export_csv=export_csv)
                else: # no face found
                    save_new_face(frame)
            else: # not image in dir
                save_new_face(frame)

            # finish processing the frame
            fq.task_done()
    except:
        traceback.print_exc()
    finally:
        shutdown.set()


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
    if params['shutdown'].is_set():
        exit(0)
   
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
            if ratio > params['face_bbox_region_ratio']:
                frame_crop = frame[fb[0][1]:fb[1][1], fb[0][0]:fb[1][0]].copy()

                # show tip
                params['show_tip'] = False 

                # process
                if not params['fq'].qsize():
                    params['fq'].put(frame_crop)
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
    # read config
    config = None
    with open('config.json') as f:
        config = json.load(f)

    print(json)

    # setup
    update_params = {
        'enable_debug_info': True,
        'enable_recognition': False,
        'face_region': (0.6, 0.6) if config is None else config['face_region'],
        'face_bbox': None,
        'face_bbox_region_ratio': 0.5 if config is None else config['face_bbox_region_ratio'],
        'db_path': './imgs',
        'deepface_model': DEEPFACE_MODELS[1],
        'timeout': 0.25 if config is None else config['timeout'],
        'repeat_log_s': 5 if config is None else config['repeat_log_s'],
        'show_tip': False, 
        'fq': Queue(),
        'shutdown': Event(),
    }

    # startup processing server
    proc_pid = Thread(target=server_face_processing, args=(
        update_params['fq'], 
        update_params['shutdown'],
        update_params['db_path'],
        update_params['deepface_model'],
        update_params['repeat_log_s'],
        update_params['timeout'],
    ))
    proc_pid.start()

    # run cv (camera read only)
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

    # shutdown
    update_params['shutdown'].set()
    proc_pid.join()


