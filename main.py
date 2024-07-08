import os
import sys
sys.path.append("./yolo_v8_human")
import copy
import glob

import cv2
from joblib import Parallel,delayed
from natsort import natsorted
from tqdm import tqdm
import jsonlines

from tools.yolo_v8_cpu import YOLO_V8_N
from tools.mediapipe_hpe import MP_HPE
from tools.multi_show import show_multi_imgs

class M2DHPE(object):
    def __init__(self) -> None:
        self.yolov8n=YOLO_V8_N()
        self.mphpe=MP_HPE()
    def run(self,frame,vis=False):
        outputs=self.yolov8n.run(
            frame=copy.deepcopy(frame),
            vis=vis,
            conf_threshold=0.5,
            iou_threshold=0.5
        ) 
        # import pdb;pdb.set_trace()
        results=[]
        for output in outputs:
            x1, y1, x2, y2, score, index = output
            sub_frame=frame[int(y1):int(y2),int(x1):int(x2),:]
            joints=self.mphpe.run(frame=copy.deepcopy(sub_frame),vis=vis)
            # align pixel coord
            for key in joints:
                joints[key]['x']+=x1
                joints[key]['y']+=y1
            results.append(joints)
        bboxs,hpes=outputs,results
        return bboxs,hpes
    def vis(self,frame,bboxs,hpes):
        image=copy.deepcopy(frame)
        for bbox in bboxs:
            x1, y1, x2, y2, score, index = bbox
            cv2.rectangle(
                img=image,
                pt1=(int(x1),int(y1)),
                pt2=(int(x2),int(y2)),
                color=(0,0,255),
                thickness=5
            )
            # import pdb;pdb.set_trace()
        for result in hpes:
            joints=result
            for key in joints:
                joint=joints[key]
                x=int(joint['x'])
                y=int(joint['y'])
                image=cv2.circle(
                    img=image,
                    center=(x,y),
                    radius=2,
                    color=(0,255,0),
                    thickness=-1
                )
        return image

def get_cam_lists(root_path,cam_num):
    temps=[]
    for i in range(cam_num):
        temp=natsorted(glob.glob(os.path.join(root_path,f"cam{i+1}","*")))
        temps.append(temp)
    min_len=len(temps[0])
    for temp in temps:
        if len(temp)<min_len:
            min_len=len(temp)
    min_len-=5
    lists=[]
    for i in range(min_len):
        list=[]
        for j in range(cam_num):
            list.append(temps[j][i].replace('\\','/'))
        lists.append(list)
    return lists

def test():
    mmpe=M2DHPE()
    frame=cv2.imread("./test/snow51.jpg")
    bboxs,hpes=mmpe.run(frame=frame)
    annotated_frame=mmpe.vis(frame,bboxs,hpes)
    cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("image",annotated_frame)
    cv2.waitKey(0)

def infer(mmpe,frame):
    bboxs,hpes=mmpe.run(frame)
    annotated_frame=mmpe.vis(frame,bboxs,hpes)
    return annotated_frame,bboxs,hpes

def main():
    root_path="./snow51"
    cam_num=6
    mmpes=[M2DHPE() for _ in range(cam_num)]
    frame_lists=get_cam_lists(root_path=root_path,cam_num=cam_num)
    cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
    with jsonlines.open(os.path.join(root_path,"results.json"),'w') as writer:
        for frame_list in tqdm(frame_lists):
            results=Parallel(n_jobs=cam_num,backend="threading")(
                delayed(infer)(mmpe,cv2.imread(frame_path))
                for frame_path,mmpe in list(zip(frame_list,mmpes))
            )
            annotated_frame_list,bboxs_list,hpes_list=zip(*results)
            obj={
                "hpe":list(hpes_list),
                "bbox":list(bboxs_list),
                "frame":list(frame_list)
            }
            writer.write(obj)
            # import pdb;pdb.set_trace()
            total_frame=show_multi_imgs(
                scale=1,
                imglist=list(annotated_frame_list),
                order=(2,3)
            )
            cv2.imshow("image",total_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break



if __name__=="__main__":
    main()
