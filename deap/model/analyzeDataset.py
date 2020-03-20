import cv2
import os
from joblib import Parallel, delayed
from joblib import parallel_backend
video_path = "/work/aashfaq/datasets/deap/videos/"
image_path = "/work/aashfaq/datasets/deap/images/"


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_frames(full_path, video_name):
    im_save_folder = image_path + video_name.split(".")[0] + "/"
    make_folder(im_save_folder)
    vidcap = cv2.VideoCapture(full_path)
    success,image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        if count % 100 == 0:
            # print(im_save_folder+"frame%d.jpg" % count)
            cv2.imwrite(im_save_folder+"frame%d.jpg" % count, image)     # save frame as JPEG file
        count += 1

    print("images saved for ", video_name)
    return True

inputs = []
for root, subdirs, files in os.walk(video_path):
    for file in files:
        # print(root, subdirs, files)
        if len(subdirs) == 0:
            filepath = root+ "/" + file
            inputs.append((filepath, file))


with parallel_backend('multiprocessing', n_jobs=10):
    Parallel()(delayed(get_frames)(input[0], input[1]) for input in inputs)


"""
https://cmusatyalab.github.io/openface/demo-3-classifier/
get aligned open face images
for N in {1..8}; do python util/align-dlib.py /work/aashfaq/datasets/deap/images align outerEyesAndNose /work/aashfaq/datasets/deap/images_aligned/ --size 96 & done


batch-represent/main.lua -outDir /work/aashfaq/datasets/deap/images_features/  -data /work/aashfaq/datasets/deap/images_aligned/


./demos/classifier.py train /work/aashfaq/datasets/deap/images_features/
"""