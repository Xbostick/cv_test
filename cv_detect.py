import os
import cv2


import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from ultralytics import YOLO

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

side_bounds = [
    np.array([[98,716], [924,411], [1130, 467], [403,873]]),
    np.array([[669,981], [1364,426], [1641,451], [1356,1074]]), 
]

passing_bounds = [
     np.array([[0,771], [120,698], [437,838], [0,1115]]),
    np.array([[0,1286], [544,880], [1580,1021], [1520,1508]]),

]

barrier_state = {
    'open' : 1,
    'close' : 0
    }
class barrier():
    def __init__(self, coordinates):
        self.state = barrier_state["close"]
        self.coordinates = coordinates
    def show(self, img):
        if self.state == barrier_state["close"]:
            cv2.line(img, self.coordinates[0], self.coordinates[1],(255, 0, 0), 9)

left_barier = barrier(((20,677),(411,856)))
right_barier = barrier(((376,967),(1576,1123)))

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
# 	print(A.x,B.x,C.x,D.x)
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def check_cross(img, rect, poly, corner = None, need_points = False):
    rect_t_l = Point(*rect[0:2])
    rect_b_r = Point(*rect[2:4])
    rect_t_r = Point(rect_b_r.x,rect_t_l.y)
    rect_b_l = Point(rect_t_l.x,rect_b_r.y)
    rect_points = [ rect_b_l, rect_t_l, rect_t_r, rect_b_r,]
    poly_points = [Point(x,y) for x,y in poly]
    cross = 0

    if corner == "br":
        if rect_b_r.x > poly_points[2].x and rect_b_r.y > poly_points[2].y:
            cv2.circle(img,(rect_b_r.x,rect_b_r.y), 10, (255,0,255),20)
            cv2.circle(img,(poly_points[2].x,poly_points[2].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2, rect_point_1, rect_point_2 in zip(poly_points[2:], [poly_points[3],poly_points[0]], rect_points[2:], [rect_points[3],rect_points[0]]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (0,0,255),20)
                    cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
                    cross += 1
    elif corner == "tl":
        if rect_b_l.x < poly_points[1].x and rect_b_l.y < poly_points[1].y:
            cv2.circle(img,(rect_b_r.x,rect_b_r.y), 10, (255,0,255),20)
            cv2.circle(img,(poly_points[1].x,poly_points[1].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2 in zip(poly_points[:2], poly_points[1:]):
            for rect_point_1, rect_point_2 in zip(rect_points[:2],rect_points[1:]):
                cross += intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2)
    elif corner == "bl":
        if rect_t_l.x < poly_points[0].x and rect_t_l.y < poly_points[0].y:
            cv2.circle(img,(rect_t_l.x,rect_t_l.y), 10, (255,0,255),20)
            cv2.circle(img,(poly_points[1].x,poly_points[1].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2 in zip([poly_points[3], poly_points[0]], poly_points[:2]):
            for rect_point_1, rect_point_2 in zip([rect_points[3], rect_points[0]],rect_points[:2]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (0,0,255),20)
                    cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
                    cross += 1
    else:
        for poly_point_1, poly_point_2 in zip(poly_points, poly_points[1:]+ [poly_points[0]]):
            for rect_point_1, rect_point_2 in zip(rect_points, rect_points[1:] + [rect_points[0]]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    cross += 1
                    # cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (255,0,255),20)
                    # cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
    if need_points:
        return [cross, rect_points, poly_points]
    else:
        return cross



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files



def process_video(path, state_shared_memory = None, verbose = 3, save_signals = True, auto_mode = True):
    frame_cars_left = 0
    frame_cars_right = 0
    frame_passing_cars_left = 0
    frame_passing_cars_right = 0
    signals_right = 0
    signals_left = 0
    is_right_car_passed = False
    is_left_car_passed = False
    is_right_car_passing = False
    is_left_car_passing= False
    left_timer = 0
    right_timer = 0
    barrier_delay = 5
    is_avalible_to_ride_left = False
    is_avalible_to_ride_right = True

    data = LoadImages(path) # itterable
    model = YOLO("yolov5su.pt")

    for i,one in enumerate(data):
        name = one[0]
        img = one[2]

        predictions = model.predict(img, classes = [2,7,3], conf = 0.25)

        for pred in predictions:
            if len(pred.boxes.cls):
                for rec in pred.boxes.xyxy:
                    rec = rec.cpu().numpy().astype(int)
                    left_cross,rect_points,poly_points = check_cross(img,rec, side_bounds[0], need_points=True)
                    rect_points = rect_points + [rect_points[0]]      
                    poly_points = poly_points + [poly_points[0]]
                    if verbose > 3:             
                        for p in range(4):
                            cv2.line(img,(rect_points[p].x,rect_points[p].y), (rect_points[p+1].x, rect_points[p+1].y), (70*p, 0, 255 - 70*p), 20)
                            cv2.line(img,(poly_points[p].x,poly_points[p].y), (poly_points[p+1].x, poly_points[p+1].y), (70*p,0,255 - 70*p), 20)

                    right_cross,rect_points,poly_points = check_cross(img,rec, side_bounds[1], need_points=True)
                    rect_points = rect_points + [rect_points[0]]      
                    poly_points = poly_points + [poly_points[0]]  
                    if verbose > 3:           
                        for p in range(4):
                            cv2.line(img,(rect_points[p].x,rect_points[p].y), (rect_points[p+1].x, rect_points[p+1].y), (70*p, 0, 255 - 70*p), 20)
                            cv2.line(img,(poly_points[p].x,poly_points[p].y), (poly_points[p+1].x, poly_points[p+1].y), (70*p,0,255 - 70*p), 20)

                    if ((right_cross >= 3) and (left_cross < right_cross)) or check_cross(img,rec, side_bounds[1], 'br'):
                        frame_cars_right += 1
                        if verbose:
                            cv2.putText(img, f"Car rightside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,0),5)
                        if verbose >= 2:
                            cv2.putText(img, f"Car right cross {right_cross}", (rec[2],rec[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.putText(img, f"Car left cross {left_cross}", (rec[2],rec[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    elif  left_cross >= 3 or check_cross(img,rec, side_bounds[0], 'bl'):
                        frame_cars_left += 1
                        if verbose:
                            cv2.putText(img, f"Car leftside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,0),5)
                        if verbose > 1:
                            cv2.putText(img, f"Car right cross {right_cross}", (rec[2],rec[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.putText(img, f"Car left cross {left_cross}", (rec[2],rec[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    for ind, bound_check in enumerate(passing_bounds):
                        if check_cross(img,rec, bound_check) > 4:
                            if ind == 1:
                                    frame_passing_cars_right +=1
                                    if verbose:
                                        cv2.putText(img, "Passing car rightside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                        cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,200),5)
                            elif ind == 0:
                                frame_passing_cars_left +=1
                                if verbose:
                                    cv2.putText(img, "Passing car leftside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                    cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,200),5)                            
            
            if verbose >1:
                for bound in side_bounds:
                    cv2.polylines(img,[np.int32(bound)], True,(0,255,0))
                for bound in passing_bounds:
                    cv2.polylines(img,[np.int32(bound)], True,(0,255,200))

        

        if auto_mode:
            # Если машина подьезжает - открываем
            if frame_cars_left > 100:
                signals_left += 1
                cv2.imwrite(f"./photos/signal_left_{signals_left}.jpg", img)
                frame_cars_left = 0
                left_barier.state = barrier_state['open']
                left_timer = time.time()
            if frame_cars_right > 100:
                signals_right += 1
                cv2.imwrite(f"./photos/signal_right_{signals_right}.jpg", img)
                frame_cars_right = 0
                right_barier.state = barrier_state['open']
                right_timer = time.time()

            # Если машина в статусе проезжает(passing), но стоит долго, а шлагбаум закрыт - поднимаем
            if frame_passing_cars_left > 100 and left_barier.state == barrier_state['close']:
                signals_left += 1
                if save_signals:
                    cv2.imwrite(f"./photos/signal_left_{signals_left}.jpg", img)
                frame_passing_cars_left = 0
                left_barier.state = barrier_state['open']
                left_timer = time.time()
            if frame_passing_cars_right > 100 and right_barier.state == barrier_state['close']:
                signals_right += 1
                if save_signals:
                    cv2.imwrite(f"./photos/signal_right_{signals_right}.jpg", img)
                frame_passing_cars_right = 0
                right_barier.state = barrier_state['open']
                right_timer = time.time()
            
            # Если истек таймер и проезжающих машин нет - закрываем шлагбаум

            left_barier.show(img)
            right_barier.show(img)

            if time.time() - left_timer > barrier_delay:
                left_timer = 0
                left_barier.state = barrier_state['close']
            if time.time() - right_timer  > barrier_delay:
                right_timer = 0
                right_barier.state = barrier_state['close']

            if state_shared_memory:
                state_shared_memory[0] = left_barier.state 
                state_shared_memory[1] = right_barier.state
        
            
            cv2.putText(img, f"Left_signals = {signals_left}, Right_signals = {signals_right}", (1800,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        
        cv2.imshow(one[0], one[2])
        
        cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    process_video("cvtest.avi")