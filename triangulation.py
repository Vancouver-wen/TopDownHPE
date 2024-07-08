import os
import sys
sys.path.append("./")
import json
from itertools import compress
import gc
from io import StringIO
import io

import cv2
import numpy as np 
import jsonlines
from natsort import natsorted
from tqdm import tqdm
from sklearn.cluster import DBSCAN,KMeans,MeanShift,AffinityPropagation
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,SequentialSampler
import PIL

from tools.instance_level_cost_matrix import get_instance_level_cost
from tools.disjoint_set_cluster import DisjointSetCluster
from tools.iterative_maximum_clique import IterativeMaximunCalique
from tools.instance_level_triangulation import instance_level_clique_triangulate
from tools.multi_show import show_multi_imgs

joint_names=[
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
    'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]
joint_weights=[
    1,0,1,0,0,1,0,
    1,1,0,0,
    1,1,1,1,1,1,
    0,0,0,0,0,0,
    1,1,
    1,1,1,1,
    0,0,0,0
]

def vis_instances(instances,labels):
    frames_map=dict()
    for instance,label in list(zip(instances,labels)):
        cam_index=instance['camera']
        frame_path=instance['frame_path']
        frame=frames_map.setdefault(cam_index,cv2.imread(frame_path))
        bbox=instance['bbox']
        x1, y1, x2, y2, score, index=bbox
        frame=cv2.rectangle(
            img=frame,
            pt1=(int(x1),int(y1)),
            pt2=(int(x2),int(y2)),
            color=(255,0,0),
            thickness=2
        )
        joints=instance['joints']
        for joint in joints:
            x=int(joint[0])
            y=int(joint[1])
            frame=cv2.circle(
                img=frame,
                center=(x,y),
                radius=4,
                color=(0,0,255),
                thickness=-1
            )
        frame=cv2.putText(
            img=frame,
            text=f"{int(label)}",
            org=(int(x1),int(y1)+50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=2,
            color=(0,255,0),
            thickness=4

        )
    frames=[]
    for key in natsorted(frames_map.keys()):
        frame=frames_map[key]
        frames.append(frame)
    total_frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(2,3)
    )
    # cv2.imwrite("./test/vis_instance.jpg",total_frame)
    return total_frame

def generate_hpe_cost_matrix(
        frames,
        bboxes,
        poses,
        calibrations
    ):
    assert len(frames)==len(calibrations),f"require len(frames)==len(calibrations)"
    assert len(bboxes)==len(calibrations),f"require len(bboxes)==len(calibrations)"
    assert len(poses)==len(calibrations),f"require len(hpes)==len(calibrations)"
    instances=[]
    for step,(frame,bboxs,persons,calibration) in enumerate(list(zip(frames,bboxes,poses,calibrations))):
        for bbox,person in list(zip(bboxs,persons)):
            available_joint_names=list(compress(joint_names,joint_weights))
            try:
                temps=[person[available_joint_name] for available_joint_name in available_joint_names]
                instance={
                    'frame_path':frame,
                    'bbox': bbox,
                    'camera':step,
                    'calibration': calibration,
                    'joints':[[temp['x'],temp['y'],temp['vis']] for temp in temps],
                    'definitions':available_joint_names
                }
                instances.append(instance)
            except:
                print(f"empty: {person}")
    cost_matrix=np.zeros(
        shape=(len(instances),len(instances)),
        dtype=np.float64
    )
    for i in range(len(instances)):
        for j in range(len(instances)):
            cost_matrix[i][j]=get_instance_level_cost(
                instance1=instances[i],
                instance2=instances[j],
                calibrations=calibrations
            )
    np.set_printoptions(
        threshold=5000,
        linewidth=5000
    )
    print(f"cost_matrix:\n {cost_matrix}")
    # labels=DBSCAN(eps=1e2, min_samples=2, metric="precomputed").fit_predict(cost_matrix)
    # print(f"=> DBSCAN labels: {labels}")
    # total_frame_dbscan=vis_instances(
    #     instances=instances,
    #     labels=labels
    # )
    return instances,cost_matrix

def generate_clique(labels,instances):
    cliques=dict()
    for label,instance in list(zip(labels,instances)):
        if label==-1:
            continue
        cliques.setdefault(int(label),[]).append(instance)
    return cliques

def vis_human_3ds(human_3ds):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-5, 5) 
    ax.set_ylim(-5, 5) 
    ax.set_zlim(0, 5) 
    for human_3d in human_3ds:
        for joint_name in human_3d:
            joint_3d=human_3d[joint_name]
            ax.scatter3D(joint_3d[0],joint_3d[1],joint_3d[2],s=1,color='blue')
    #申请缓冲地址
    buffer_ = io.BytesIO()#using buffer,great way!
    #保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    plt.savefig(buffer_,dpi=1000,format='jpg')
    ax.remove()
    fig.clear()
    plt.close(fig)
    del fig, ax
    gc.collect()
    buffer_.seek(0)
    #用PIL或CV2从内存中读取
    dataPIL = PIL.Image.open(buffer_)
    #转换为nparrary，PIL转换就非常快了,data即为所需
    data = np.asarray(dataPIL)
    # cv2.imshow('image', data)
    #释放缓存    
    buffer_.close()
    return data

class MyDataset(Dataset):
    def __init__(self,json_path,calibrations):
        self.objs=[]
        with jsonlines.open(json_path,'r') as reader:
            for obj in reader:
                self.objs.append(obj)
        self.calibrations=calibrations
    
    def __getitem__(self, index):
        obj=self.objs[index]
        hpes=obj['hpe']
        bboxes=obj['bbox'] # x1, y1, x2, y2, score, index
        frames=obj['frame']
        instances,cost_matrix=generate_hpe_cost_matrix(
            frames=frames,
            bboxes=bboxes,
            poses=hpes,
            calibrations=self.calibrations
        )
        # labels=DisjointSetCluster(eps=1e3,min_samples=3).fit_predict(cost_matrix)
        # print(f"=> DisjointSetCluster labels: {labels}")
        labels=IterativeMaximunCalique(eps=1e2,min_samples=3).fit_predict(cost_matrix)
        print(f"=> IterativeMaximunCalique labels: {labels}")
        total_frame_cluster=vis_instances(
            instances=instances,
            labels=labels
        )
        cliques=generate_clique(labels,instances)
        human_3ds=[]
        for key in cliques:
            clique=cliques[key]
            human_3d=instance_level_clique_triangulate(clique=clique)
            human_3ds.append(human_3d)
        scene_3d=vis_human_3ds(human_3ds)
        ratio=total_frame_cluster.shape[0]/scene_3d.shape[0]
        scene_3d=cv2.resize(scene_3d,dsize=None,fx=ratio,fy=ratio)
        # import pdb;pdb.set_trace()
        frame=np.hstack((total_frame_cluster,scene_3d))
        frame=cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
        return index,human_3ds,frame

    def __len__(self):
        return len(self.objs)

def main():
    calibration_path="./snow51/cameras.json"
    calibrations=[]
    with jsonlines.open(calibration_path,'r') as reader:
        for obj in reader:
            calibrations.append(obj)

    pbar=tqdm(total=24970)
    json_path="./snow51/results.json"
    save_path="./snow51/human_3ds.json"
    out=None
    myDataset=MyDataset(json_path,calibrations)
    myDataloader=DataLoader(
        dataset=myDataset,
        batch_size=1,
        shuffle=False,
        num_workers=24,
        collate_fn=lambda x:x
    )
    with jsonlines.open(save_path,'w') as writer:
        for batch in myDataloader:
            index,human_3ds,frame=batch[0]
            print(f'=> index:{index}')
            writer.write(human_3ds)
            cv2.imwrite(f"./test/test/vis_instance_v3.jpg",frame)
            # import pdb;pdb.set_trace()
            height,width,channel=frame.shape
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('./test/test/vis_instance_v3.avi',fourcc, 25.0,(width,height),True)
                # out = cv2.VideoWriter('./test/test/vis_instance_v2.mp4',cv2.VideoWriter_fourcc('H', '2', '6', '4'),25,(width,height))
                print("=> create video writer")
            out.write(frame)
            pbar.update(1) 
    if out is not None:
        out.release()
if __name__=="__main__":
    main()