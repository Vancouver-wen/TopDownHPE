import os
import sys

import mediapipe
import cv2

class MP_HPE(object):
    def __init__(self) -> None:
        self.mp_pose = mediapipe.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            # enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mediapipe.solutions.drawing_utils
        self.skeleton_definiton=self.get_skeleton_definition()

    def get_skeleton_definition(self,):
        skeleton_definition=[
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        return skeleton_definition
    
    def run(
            self,
            frame,
            vis=False
        ):
        """
        info: human detect
        params:
          - frame: numpy image
          - vis: whether cv2.imshow
        return:
          - outputs: list of (x1, y1, x2, y2, score, index)
        
        """
        img=frame
        # read img BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size=img.shape[:2]
        height,width=img_size
    
        results = self.pose.process(img)
        
        if vis:
            cv2.namedWindow("keypoint",cv2.WINDOW_GUI_NORMAL)
            self.drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.imshow("keypoint", img)
            self.drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # format results
        outputs=dict()
        if results.pose_landmarks is None:
            # cv2.imshow("frame",img)
            # cv2.waitKey(0)
            return outputs
        for joint_type,output in list(zip(self.skeleton_definiton,results.pose_landmarks.landmark)):
            outputs[joint_type]={
                'x': output.x*width,
                'y': output.y*height,
                'z': output.z*(width+height)/2,
                'vis': output.visibility
            }
        if vis:
            for output in outputs:
                temp=outputs[output]
                x=temp['x']
                y=temp['y']
                img=cv2.circle(
                    img=img,
                    center=(int(x),int(y)),
                    radius=2,
                    color=(0,0,255),
                    thickness=-1
                )
            cv2.imshow("img",img)
            cv2.waitKey(0)
        return outputs
