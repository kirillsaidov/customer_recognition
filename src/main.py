# module main

from deepface import DeepFace

models = [
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


if __name__ == '__main__':
    print('starting...')

    dfs = DeepFace.find(img_path = './imgs/img_id1.jpg', db_path = './imgs', model_name=models[1])
    print('recognized:', dfs)

    dfs = DeepFace.find(img_path = './imgs/img_id2.jpg', db_path = './imgs', model_name=models[1])
    print('recognized:', dfs)

    dfs = DeepFace.find(img_path = './imgs/img_id1.jpg', db_path = './imgs', model_name=models[1])
    print('recognized:', dfs)


    print('end.')




