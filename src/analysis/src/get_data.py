import pandas as pd
import cv2
import re
import numpy
import pickle
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import Assembler


#file_path = ''

#df = pd.read_csv(file_path)

file_path = './sample.pickle'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

#print(data)
data = dict(
    data
)

header = data.pop("metadata")
print(header)

all_jointnames = header["all_joints_names"]
print(all_jointnames)

numjoints = len(all_jointnames)
bpts = range(numjoints)

print(bpts)

frame_names = list(data)
print(frame_names)

frames = [int(re.findall(r"\d+", name)[0]) for name in frame_names]
print(frames)

ind = frames.index(1)
dets = Assembler._flatten_detections(data[frame_names[ind]])

print(dets)

for det in dets:
    print(det.label)
    print(det.pos)
