import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple
import time
import glob

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


# Define regions for car detection
SIDE_BOUNDS = [
    np.array([[98, 716], [924, 411], [1130, 467], [403, 873]]),  # Left side
    np.array([[669, 981], [1364, 426], [1641, 451], [1356, 1074]]),  # Right side
]

PASSING_BOUNDS = [
    np.array([[0, 771], [120, 698], [437, 838], [0, 1115]]),  # Left passing
    np.array([[0, 1286], [544, 880], [1580, 1021], [1520, 1508]]),  # Right passing
]

BARRIER_STATE = {
    'open': 1,
    'close': 0
}


class Barrier:
    """Represents a barrier with open/close state and visualization."""
    def __init__(self, coordinates: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.state = BARRIER_STATE["close"]
        self.coordinates = coordinates

    def show(self, img: np.ndarray):
        """Draw the barrier on the image if closed."""
        if self.state == BARRIER_STATE["close"]:
            cv2.line(img, self.coordinates[0], self.coordinates[1], (255, 0, 0), 9)

# Initialize barriers
LEFT_BARRIER = Barrier(((20, 677), (411, 856)))
RIGHT_BARRIER = Barrier(((376, 967), (1576, 1123)))

class Point:
    """Represents a 2D point for intersection calculations."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def ccw(A: Point, B: Point, C: Point) -> bool:
    """Check if points A, B, C are in counter-clockwise order."""
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """Check if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)



def check_cross(
    img: np.ndarray,
    rect: np.ndarray,
    poly: np.ndarray,
    corner: str = None,
    need_points: bool = False,
    verbose: int = 2
) -> int:
    """
    Check if a rectangle intersects with a polygon.
    
    Args:
        img: Image to draw debug information on
        rect: Rectangle coordinates [x1, y1, x2, y2]
        poly: Polygon coordinates
        corner: Specific corner to check ('br', 'tl', 'bl')
        need_points: Whether to return points
        verbose: Verbosity level for debugging
        
    Returns:
        Number of intersections or tuple with points if need_points=True
    """
    assert len(rect) == 4, "Rectangle must have 4 coordinates"
    assert len(poly) >= 3, "Polygon must have at least 3 points"
    rect_t_l = Point(*rect[0:2])
    rect_b_r = Point(*rect[2:4])
    rect_t_r = Point(rect_b_r.x,rect_t_l.y)
    rect_b_l = Point(rect_t_l.x,rect_b_r.y)
    rect_points = [ rect_b_l, rect_t_l, rect_t_r, rect_b_r,]
    poly_points = [Point(x,y) for x,y in poly]
    cross = 0

    if corner == "br":
        if rect_b_r.x > poly_points[2].x and rect_b_r.y > poly_points[2].y:
            if verbose > 2:
                cv2.circle(img,(rect_b_r.x,rect_b_r.y), 10, (255,0,255),20)
                cv2.circle(img,(poly_points[2].x,poly_points[2].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2, rect_point_1, rect_point_2 in zip(poly_points[2:], [poly_points[3],poly_points[0]], rect_points[2:], [rect_points[3],rect_points[0]]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    if verbose > 2:
                        cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (0,0,255),20)
                        cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
                    cross += 1
    elif corner == "tl":
        if rect_b_l.x < poly_points[1].x and rect_b_l.y < poly_points[1].y:
            if verbose > 2:
                cv2.circle(img,(rect_b_r.x,rect_b_r.y), 10, (255,0,255),20)
                cv2.circle(img,(poly_points[1].x,poly_points[1].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2 in zip(poly_points[:2], poly_points[1:]):
            for rect_point_1, rect_point_2 in zip(rect_points[:2],rect_points[1:]):
                cross += intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2)
    elif corner == "bl":
        if rect_t_l.x < poly_points[0].x and rect_t_l.y < poly_points[0].y:
            if verbose > 2:
                cv2.circle(img,(rect_t_l.x,rect_t_l.y), 10, (255,0,255),20)
                cv2.circle(img,(poly_points[1].x,poly_points[1].y),10,(0,0,255),20)
            return 1
        for poly_point_1, poly_point_2 in zip([poly_points[3], poly_points[0]], poly_points[:2]):
            for rect_point_1, rect_point_2 in zip([rect_points[3], rect_points[0]],rect_points[:2]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    if verbose > 2:
                        cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (0,0,255),20)
                        cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
                    cross += 1
    else:
        for poly_point_1, poly_point_2 in zip(poly_points, poly_points[1:]+ [poly_points[0]]):
            for rect_point_1, rect_point_2 in zip(rect_points, rect_points[1:] + [rect_points[0]]):
                if intersect(poly_point_1, poly_point_2, rect_point_1, rect_point_2):
                    cross += 1
                    if verbose > 3:
                        cv2.line(img, (poly_point_1.x, poly_point_1.y),  (poly_point_2.x, poly_point_2.y), (255,0,255),20)
                        cv2.line(img, (rect_point_1.x, rect_point_1.y),  (rect_point_2.x, rect_point_2.y), (0,0,255),20)
    if need_points:
        return [cross, rect_points, poly_points]
    else:
        return cross

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True, verbose=1):
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
        self.verbose = verbose > 1
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
            ret_val, img = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img = self.cap.read()

            self.frame += 1
            if self.verbose:
                print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            if self.verbose:
                print(f'image {self.count}/{self.nf} {path}: ', end='')


        return path, img, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files




def process_video(
    path: str,
    state_shared_memory=None,
    frame_queue=None,
    stop_signal=None,
    verbose: int = 1,
    save_signals: bool = True,
    mode: str = 'signals'
):
    """
    Process video to detect cars and control barriers.
    
    Args:
        path: Path to video file
        state_shared_memory: Shared array for car detection state
        frame_queue: Queue for storing processed frames
        stop_signal: Signal to stop processing
        verbose: Verbosity level
        save_signals: Whether to save signal images
        mode: Operation mode ('signals' or 'auto')
    """
    frame_cars_left = 0
    frame_cars_right = 0
    frame_passing_cars_left = 0
    frame_passing_cars_right = 0
    signals_right = 0
    signals_left = 0
    left_timer = 0
    right_timer = 0
    barrier_delay = 5


    data = LoadImages(path, verbose = verbose > 1)
    model = YOLO("yolov5su.pt", verbose=verbose > 1)

    for name, img, _ in data:
        frame_cars_left = max(0,frame_cars_left - 1)
        frame_cars_right = max(0,frame_cars_right - 1)

        predictions = model.predict(img, classes = [2,7,3], conf = 0.25,  verbose= verbose > 1)

        for pred in predictions:
            if len(pred.boxes.cls):
                for rec in pred.boxes.xyxy:
                    rec = rec.cpu().numpy().astype(int)
                    left_cross,rect_points,poly_points = check_cross(img,rec, SIDE_BOUNDS[0], need_points=True)
                    rect_points = rect_points + [rect_points[0]]      
                    poly_points = poly_points + [poly_points[0]]
                    if verbose > 3:             
                        for p in range(4):
                            cv2.line(img,(rect_points[p].x,rect_points[p].y), (rect_points[p+1].x, rect_points[p+1].y), (70*p, 0, 255 - 70*p), 20)
                            cv2.line(img,(poly_points[p].x,poly_points[p].y), (poly_points[p+1].x, poly_points[p+1].y), (70*p,0,255 - 70*p), 20)

                    right_cross,rect_points,poly_points = check_cross(img,rec, SIDE_BOUNDS[1], need_points=True)
                    rect_points = rect_points + [rect_points[0]]      
                    poly_points = poly_points + [poly_points[0]]  
                    if verbose > 3:           
                        for p in range(4):
                            cv2.line(img,(rect_points[p].x,rect_points[p].y), (rect_points[p+1].x, rect_points[p+1].y), (70*p, 0, 255 - 70*p), 20)
                            cv2.line(img,(poly_points[p].x,poly_points[p].y), (poly_points[p+1].x, poly_points[p+1].y), (70*p,0,255 - 70*p), 20)

                    if ((right_cross >= 3) and (left_cross < right_cross)) or check_cross(img,rec, SIDE_BOUNDS[1], 'br'): 
                        frame_cars_right = min(15,frame_cars_right + 2)
                        if verbose:
                            cv2.putText(img, f"Car rightside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,0),5)
                        if verbose >= 2:
                            cv2.putText(img, f"Car right cross {right_cross}", (rec[2],rec[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.putText(img, f"Car left cross {left_cross}", (rec[2],rec[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    elif  left_cross >= 3 or check_cross(img,rec, SIDE_BOUNDS[0], 'bl'):
                        frame_cars_left = min(15,frame_cars_left + 2)
                        if verbose:
                            cv2.putText(img, f"Car leftside", (rec[2],rec[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.rectangle(img,rec[0:2],rec[2:4], (0,255,0),5)
                        if verbose > 1:
                            cv2.putText(img, f"Car right cross {right_cross}", (rec[2],rec[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.putText(img, f"Car left cross {left_cross}", (rec[2],rec[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    if mode == 'auto':
                        for ind, bound_check in enumerate(PASSING_BOUNDS):
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
            
            if verbose > 1:
                for bound in SIDE_BOUNDS:
                    cv2.polylines(img,[np.int32(bound)], True,(0,255,0))
                for bound in PASSING_BOUNDS:
                    cv2.polylines(img,[np.int32(bound)], True,(0,255,200))

        if mode == 'signals':
            if verbose: 
                cv2.putText(img, f"Model Yolov5_su",(100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(img, f"Is_car_leftside = {frame_cars_left > 10}, is car rightside = {frame_cars_right > 10}",(1800,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if verbose > 1:
                cv2.putText(img, f"Left_frames = {frame_cars_left}, Right_frames = {frame_cars_right}", (1800,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
            state_shared_memory[0] = int(frame_cars_left > 10)
            state_shared_memory[1] = int(frame_cars_right > 10)
        

        if mode == "auto":
            # Если машина подьезжает - открываем
            if frame_cars_left > 10:
                signals_left += 1
                cv2.imwrite(f"./photos/signal_left_{signals_left}.jpg", img)
                LEFT_BARRIER.state = BARRIER_STATE['open']
                left_timer = time.time()
            if frame_cars_right > 10:
                signals_right += 1
                cv2.imwrite(f"./photos/signal_right_{signals_right}.jpg", img)
                LEFT_BARRIER.state = BARRIER_STATE['open']
                right_timer = time.time()

            # Если машина в статусе проезжает(passing), но стоит долго, а шлагбаум закрыт - поднимаем
            if frame_passing_cars_left > 50 and LEFT_BARRIER.state == BARRIER_STATE['close']:
                signals_left += 1
                if save_signals:
                    cv2.imwrite(f"./photos/signal_left_{signals_left}.jpg", img)
                frame_passing_cars_left = 0
                LEFT_BARRIER.state = BARRIER_STATE['open']
                left_timer = time.time()
            if frame_passing_cars_right > 50 and RIGHT_BARRIER.state == BARRIER_STATE['close']:
                signals_right += 1
                if save_signals:
                    cv2.imwrite(f"./photos/signal_right_{signals_right}.jpg", img)
                frame_passing_cars_right = 0
                RIGHT_BARRIER.state = BARRIER_STATE['open']
                right_timer = time.time()
            
            # Если истек таймер и проезжающих машин нет - закрываем шлагбаум

            if time.time() - left_timer > barrier_delay:
                left_timer = 0
                LEFT_BARRIER.state = BARRIER_STATE['close']
            if time.time() - right_timer  > barrier_delay:
                right_timer = 0
                RIGHT_BARRIER.state = BARRIER_STATE['close']

            LEFT_BARRIER.show(img)
            RIGHT_BARRIER.show(img)

            if state_shared_memory:
                state_shared_memory[0] = LEFT_BARRIER.state 
                state_shared_memory[1] = RIGHT_BARRIER.state
        
            
        if frame_queue:
            frame_queue.put(img.copy())
        else:
            cv2.imshow(name, img)
        
        if stop_signal and stop_signal.value :
            return 0
        cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    process_video("cvtest.avi", mode= "auto")