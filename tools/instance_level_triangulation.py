import os
import sys

import numpy as np
import cv2

def get_normalized_points(point,camera):
    """
    返回归一化平面坐标系上的一点
    """
    point=cv2.undistortPoints(
        src=np.expand_dims(np.array(point),axis=0),
        cameraMatrix=np.array(camera['K']),
        distCoeffs=np.squeeze(np.array(camera['dist'])),
        # P=np.array(camera1['K'])
    )
    # import pdb;pdb.set_trace()
    return np.squeeze(point).tolist()

def multi_view_triangulate(
        point_2ds,
        poses,
        solve_method="SVD"
    ):
    assert len(point_2ds)==len(poses),"illegal reconstruction parameters"
    if len(poses)<2:
        # triangulation need atleast 2 camera views
        return None
    A=[]
    for point_2d,pose in list(zip(point_2ds,poses)):
        P_matrix=np.concatenate(
            (np.array(pose['R']),np.expand_dims(np.array(pose['t']).T,axis=1)),
            axis=1
        )
        x=point_2d[0]
        y=point_2d[1]
        A.append(x*P_matrix[2]-P_matrix[0])
        A.append(y*P_matrix[2]-P_matrix[1])
    A=np.array(A).astype(np.float32)
    if solve_method=="SVD":
        U,sigma,VH = np.linalg.svd(A,full_matrices=True)
        vector=VH[-1]
        point_3d=vector[:3]/vector[3]
    else:
        # 这个解法不好
        eigen_value,eigen_vector = np.linalg.eig(A.T@A)
        vector=eigen_vector[np.argmin(eigen_value)]
        point_3d=vector[:3]/vector[3]
    return point_3d

def get_poses_from_clique(clique:list):
    poses=[]
    for instance in clique:
        poses.append(instance['calibration'])
    return poses

def get_point_2ds_from_clique(clique:list,joint_index:int):
    point_2ds=[]
    for instance in clique:
        point_2ds.append(instance['joints'][joint_index])
    return np.array(point_2ds)[:,:2]

def instance_level_clique_triangulate(clique:list):
    joint_names=clique[0]['definitions']
    poses=get_poses_from_clique(clique=clique)
    point_3ds=dict()
    for joint_index,joint_name in enumerate(joint_names):
        point_2ds=get_point_2ds_from_clique(clique=clique,joint_index=joint_index)
        normalized_point_2ds=[
            get_normalized_points(point=point,camera=camera)
            for point,camera in list(zip(point_2ds,poses))
        ]
        point_3d=multi_view_triangulate(
            point_2ds=normalized_point_2ds,
            poses=poses
        )
        point_3ds[joint_name]=point_3d.tolist()
    return point_3ds
    


if __name__=="__main__":
    pass