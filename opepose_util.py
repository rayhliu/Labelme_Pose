#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:29:32 2019

@author: rayhliu
"""

import os
from sys import platform
import sys
from os import getcwd
import argparse

class ITRI_OP():
    def __init__(self,config=None):
        self.config = config
    
    def init_openpose(self):
        # add environment
        pose_dir = os.path.join(getcwd(),'pose','op')
        try:
            if platform == "win32":
                sys.path.append(os.path.join(pose_dir,'python','openpose','Release'))
                os.environ['PATH']  = os.environ['PATH'] + ';'+os.path.join(pose_dir,'x64','Release') + ';'+ os.path.join(pose_dir,'bin')+';'
                import pyopenpose as op
            else:
                sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: Pose library could not be found.')
            raise e
        
        # add Flags
        parser = argparse.ArgumentParser()
        params = dict()
        parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        params["num_gpu"] = '1'
        params["num_gpu_start"] = '0'
    
        # params["net_resolution"] = "1312x736"
        if self.config is None:
            params["model_pose"] = 'BODY_25B'
        else:
            if self.config["OP_MODEL"] != "BODY_25":
                params["model_pose"] = self.config["OP_MODEL"]
            
        # params["scale_number"] = "2"
        # params["scale_gap"] = "0.25"
        
        # params["net_resolution"] = [-1*480]
        if platform == "win32":
            parser.add_argument("--image_dir", default=os.path.join(pose_dir,'examples','media'), help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
            params["model_folder"] = os.path.join(pose_dir,'models')
        else:
            parser.add_argument("--image_dir", default='/data/openpose/examples/media/', help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
            params["model_folder"] = '/data/openpose/models/'
        args = parser.parse_known_args()
        
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item
        
        # Starting Pose detector
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        return op, opWrapper
    
    def convert_op_poseInfo(self,datum,SHOULDER_THRESHOLD_MIN=20,SHOULDER_THRESHOLD_MAX=100): 
        def _two_points_length(p0,p1):
            return pow(pow(p0[0]-p1[0],2)+pow(p0[1]-p1[1],2),0.5)
        
        def _process_poss_value(value):
            x = int(round(max(value[0],0)))
            y = int(round(max(value[1],0)))
            return (x,y,value[2])
        
        handpoints_dict = {}
        if len(datum.poseKeypoints.shape) == 3:
            if datum.poseScores.size ==1:
                poseScoresList = [datum.poseScores.item()]
            else:
                poseScoresList = datum.poseScores.tolist()
                
            # body index & score 
            for index , score in enumerate(poseScoresList): 
                if self.config["OP_MODEL"] in ["BODY_25B","BODY_135"]:
                    left_shoulder = datum.poseKeypoints[index][5]
                    left_elbow = datum.poseKeypoints[index][7]
                    left_wrist = datum.poseKeypoints[index][9]
                    right_shoulder = datum.poseKeypoints[index][6]
                    right_elbow = datum.poseKeypoints[index][8]
                    right_wrist = datum.poseKeypoints[index][10]
                    neck = datum.poseKeypoints[index][17]
                    nose = datum.poseKeypoints[index][0]
                    rEye = datum.poseKeypoints[index][2]
                    lEye = datum.poseKeypoints[index][1]
                    rHip = datum.poseKeypoints[index][12]
                    lHip = datum.poseKeypoints[index][11]
                    
                else:
                    left_shoulder = datum.poseKeypoints[index][5]
                    left_elbow = datum.poseKeypoints[index][6]
                    left_wrist = datum.poseKeypoints[index][7]
                    right_shoulder = datum.poseKeypoints[index][2]
                    right_elbow = datum.poseKeypoints[index][3]
                    right_wrist = datum.poseKeypoints[index][4]
                    neck = datum.poseKeypoints[index][1]
                    nose = datum.poseKeypoints[index][0]
                    rEye = datum.poseKeypoints[index][15]
                    lEye = datum.poseKeypoints[index][16]
                    rHip = datum.poseKeypoints[index][9]
                    lHip = datum.poseKeypoints[index][12]
                    
                right_palm = None
                left_palm = None
                if datum.poseKeypoints.shape[1] == 135:
                    right_palm = datum.poseKeypoints[index][45:45+20]
                    left_palm = datum.poseKeypoints[index][25:25+20]
                p_m = tuple(neck[:2])
                shoulder_len = []
                # from neck to right shoulder
                if right_shoulder[2]!=0 and  neck[2]!=0:
                    p_s = tuple(right_shoulder[:2])
                    len_s = _two_points_length(p_s,p_m)
                    shoulder_len.append(len_s)
                # from neck to left shoulder
                if left_shoulder[2]!=0 and  neck[2]!=0:
                    p_s = tuple(left_shoulder[:2])
                    len_s = _two_points_length(p_s,p_m)
                    shoulder_len.append(len_s)
                
                if len(shoulder_len) != 0:
                    if score > 0.1 and right_shoulder[2]>0 and left_shoulder[2]>0:
                        if max(shoulder_len)>= SHOULDER_THRESHOLD_MIN and max(shoulder_len) < SHOULDER_THRESHOLD_MAX:
                            handpoints_dict[index] = []
                            handpoints_dict[index].append(score) # 0:score
                            handpoints_dict[index].append(neck) # 1:neck
                            
                            handpoints_dict[index].append((right_shoulder, # 2:right_hand 
                                          right_elbow,
                                          right_wrist))
                            
                            handpoints_dict[index].append((left_shoulder, # 3:left_hand 
                                          left_elbow,
                                          left_wrist))
                            
                            handpoints_dict[index].append(nose) # 4:nose 
                            handpoints_dict[index].append(rEye) # 5:Reye 
                            handpoints_dict[index].append(lEye) # 6:Leye 
                            
                            handpoints_dict[index].append(rHip) # 7:right_hip 
                            handpoints_dict[index].append(lHip) # 8:left_hip 
                            handpoints_dict[index].append(right_palm) #  9:right_palm
                            handpoints_dict[index].append(left_palm)  # 10:left_palm
                            
        return handpoints_dict

if __name__ == "__main__":
    from util import ITRI_Camera
    
    CAMERA_NAME = 'ELP-USB3MP01H-L180'
    CAMERA_ID = 0
    IC = ITRI_Camera(CAMERA_NAME, CAMERA_ID)
    op, opWrapper = init_openpose()
    
    
