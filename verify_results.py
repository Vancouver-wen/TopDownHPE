import os
import sys

import cv2
import numpy as np 
import jsonlines
from joblib import Parallel,delayed

from tools.multi_show import show_multi_imgs

def verify_one_frame(
        hpe,
        bbox,
        frame_path
    ):
    frame=cv2.imread(frame_path)
    for each_bbox,each_hpe in list(zip(bbox,hpe)):
        x1, y1, x2, y2, score, index = each_bbox
        cv2.rectangle(
            img=frame,
            pt1=(int(x1),int(y1)),
            pt2=(int(x2),int(y2)),
            color=(0,0,255),
            thickness=2
        )
        for key in each_hpe:
            joint=each_hpe[key]
            x=int(joint['x'])
            y=int(joint['y'])
            frame=cv2.circle(
                img=frame,
                center=(x,y),
                radius=2,
                color=(0,255,0),
                thickness=-1
            )
    return frame

def main():
    json_path="./snow51/results.json"
    cv2.namedWindow("frame",cv2.WINDOW_GUI_NORMAL)
    with jsonlines.open(json_path,'r') as reader:
        for obj in reader:
            hpes=obj['hpe']
            bboxs=obj['bbox']
            frames=obj['frame']
            all_frames=Parallel(n_jobs=6,backend="threading")(
                delayed(verify_one_frame)(hpe=hpe,bbox=bbox,frame_path=frame_path)
                for hpe,bbox,frame_path in list(zip(hpes,bboxs,frames))
            )
            all_frame=show_multi_imgs(scale=1,imglist=all_frames,order=(2,3))
            cv2.imshow("frame",all_frame)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break
            

if __name__=="__main__":
    main()