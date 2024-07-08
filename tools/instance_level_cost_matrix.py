import os
import sys
import math

import numpy as np
import cv2

def vis_epipolar_line():
    pass # TODO

def epipolar_distance(line,point):
    A,B,C=line[0],line[1],line[2]
    x,y=point[0],point[1]
    distance=abs(A*x+B*y+C)/math.sqrt(A**2+B**2)
    return distance

def symmetrix_epipolar_distance(point2,F,point1):
    # 对极约束 p2.T@F@p1=0 p(3,1) F(3,3)
    # 点在直线上
    line2=np.squeeze(cv2.computeCorrespondEpilines(
        points=np.expand_dims(point1,axis=0),
        whichImage=1,
        F=F
    ))
    line1=np.squeeze(cv2.computeCorrespondEpilines(
        points=np.expand_dims(point2,axis=0),
        whichImage=2,
        F=F
    ))
    # import pdb;pdb.set_trace()
    distance1=epipolar_distance(line1,point1)
    distance2=epipolar_distance(line2,point2)
    distance=distance1+distance2
    return distance

def get_fundamental_matrix(camera2,camera1):
    "从cam2到cam1的 Fundamental Matrix"
    K1,R1,t1=np.array(camera1['K']),np.array(camera1['R']),np.array(camera1['t'])
    K2,R2,t2=np.array(camera2['K']),np.array(camera2['R']),np.array(camera2['t'])
    R21=R2@np.linalg.inv(R1)
    t21=(t2.T-R21@t1.T).T
    t21_antisymmetric=np.array([
        [      0,  -t21[2],    t21[1]],
        [ t21[2],        0,   -t21[0]],
        [-t21[1],   t21[0],         0]
    ])
    F21=np.linalg.inv(K2).T@t21_antisymmetric@R21@np.linalg.inv(K1)
    # import pdb;pdb.set_trace()
    return F21

def get_undistort_points(point1,point2,camera1,camera2):
    """
    返回去畸变后的像素坐标系的点
    """
    point1=cv2.undistortPoints(
        src=np.expand_dims(np.array(point1),axis=0),
        cameraMatrix=np.array(camera1['K']),
        distCoeffs=np.squeeze(np.array(camera1['dist'])),
        P=np.array(camera1['K'])
    )
    point2=cv2.undistortPoints(
        src=np.expand_dims(np.array(point2),axis=0),
        cameraMatrix=np.array(camera2['K']),
        distCoeffs=np.squeeze(np.array(camera2['dist'])),
        P=np.array(camera2['K'])
    )
    # import pdb;pdb.set_trace()
    return np.squeeze(point1).tolist(),np.squeeze(point2).tolist()

def get_point_level_cost(
        point1,
        point2,
        cam_params
    ):
    if point1['cam_index']==point2['cam_index']:
        # 同一个相机内的两个点不可能在同一个簇内 -> cost 很大
        return 1e8
    camera1=cam_params[point1['cam_index']]
    camera2=cam_params[point2['cam_index']]
    # 如果 point1 与 point2 属于两个相机 -> 计算对称极限距离
    point1['point_undistorted'],point2['point_undistorted']=get_undistort_points(
        point1=point1['point_2d'],
        point2=point2['point_2d'],
        camera1=camera1,
        camera2=camera2
    )
    F=get_fundamental_matrix(
        camera2=camera2,
        camera1=camera1
    )
    distance=symmetrix_epipolar_distance(
        point2=np.array(point2['point_undistorted']),
        F=F,
        point1=np.array(point1['point_undistorted'])
    )
    # import pdb;pdb.set_trace()
    return distance

def get_instance_level_cost(
        instance1,
        instance2,
        calibrations
    ):
    costs=[]
    for joint1,joint2 in list(zip(instance1['joints'],instance2['joints'])):
        point1={
            'cam_index':instance1['camera'],
            'point_2d': joint1[:2],
            'vis': joint1[2]
        }
        point2={
            'cam_index':instance2['camera'],
            'point_2d': joint2[:2],
            'vis': joint2[2]
        }
        cost=get_point_level_cost(
            point1=point1,
            point2=point2,
            cam_params=calibrations
        )
        ratio=1/(point1['vis']*point2['vis'])
        cost*=ratio
        # try:
        #     cost=get_point_level_cost(
        #         point1=point1,
        #         point2=point2,
        #         cam_params=calibrations
        #     )
        # except:
        #     import pdb;pdb.set_trace()
        costs.append(cost)
    return np.array(costs).mean()