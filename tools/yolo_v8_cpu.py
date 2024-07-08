import os
import sys
import warnings
import argparse

import torch
import numpy as np 
import tqdm
import yaml
import cv2
from loguru import logger

from yolo_v8_human.nets import nn
from yolo_v8_human.utils import util


class YOLO_V8_N(object):
    def __init__(
            self,
            weight_path="./yolo_v8_human/weights/best.pt",
            input_size=640
        ) -> None:
        self.input_size=input_size
        self.model = nn.yolo_v8_n(num_classes=1)
        self.model = util.load_weight(weight_path, self.model)
        self.model=self.model.cpu()
        self.model.eval()
    
    @torch.no_grad()
    def run(
            self,
            frame,
            vis=False,
            conf_threshold=0.25,
            iou_threshold=0.7
        ):
        """
        info: human detect
        params:
          - frame: numpy image
          - vis: whether cv2.imshow
        return:
          - outputs: list of (x1, y1, x2, y2, score, index)
        
        """
        image = frame.copy()
        shape = image.shape[:2]

        r = self.input_size / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        height, width = image.shape[:2]

        # Scale ratio (new / old)
        r = min(1.0, self.input_size / height, self.input_size / width)

        # Compute padding
        pad = int(round(width * r)), int(round(height * r))
        w = np.mod((self.input_size - pad[0]), 32) / 2
        h = np.mod((self.input_size - pad[1]), 32) / 2

        if (width, height) != pad:  # resize
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

        # Convert HWC to CHW, BGR to RGB
        x = image.transpose((2, 0, 1))[::-1]
        x = np.ascontiguousarray(x) # 返回一个连续的array，其内存是连续的
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=0)
        x = x / 255
        # Inference
        outputs = self.model(x)
        # NMS
        outputs = util.non_max_suppression(
            outputs=outputs, 
            conf_threshold=conf_threshold, 
            iou_threshold=iou_threshold
        )
        for output in outputs:
            output[:, [0, 2]] -= w  # x padding
            output[:, [1, 3]] -= h  # y padding
            output[:, :4] /= min(height / shape[0], width / shape[1])

            output[:, 0].clamp_(0, shape[1])  # x1
            output[:, 1].clamp_(0, shape[0])  # y1
            output[:, 2].clamp_(0, shape[1])  # x2
            output[:, 3].clamp_(0, shape[0])  # y2
            if vis:
                for box in output:
                    box = box.numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if vis:
            cv2.namedWindow("Frame",cv2.WINDOW_GUI_NORMAL)
            cv2.imshow('Frame', frame)
            cv2.waitKey(10)
        # import pdb;pdb.set_trace()
        # convert to numpy
        for i in range(len(outputs)):
            outputs[i]=outputs[i].numpy()
        outputs=np.squeeze(np.array(outputs),axis=0).tolist()
        return outputs

if __name__=="__main__":
    pass