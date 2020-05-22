#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:57:42 2020

@author: rayhliu
"""

import base64
import json
import glob
import cv2
import os
from opepose_util import ITRI_OP

def run(visualize=False):
    annFolder = os.path.join('.','annotations')
    try:
        os.mkdir(annFolder)
    except OSError:
        print ("Folder: %s is existed" % annFolder)
    else:
        print ("Successfully created the directory %s " % annFolder)
    
    IOP = ITRI_OP()
    op,opWrapper = IOP.init_openpose()
    
    files = glob.glob(os.path.join('.','images','*.jpg'))
    for file in files:
        fileName = os.path.basename(file)[:-4]
        relativePath = os.path.join('..','images',fileName+'.jpg')
        frame = cv2.imread(file)
        h = frame.shape[0]
        w = frame.shape[1]
        
        
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        
        
        total_point = []
        for index,poseScore in enumerate(datum.poseScores.tolist()):
            if poseScore >= 0.3:
                poseID = str(index).zfill(2)
                for p in [[5,'ls'],[7,'le'],[9,'lw'],[6,'rs'],[8,'re'],[10,'rw'],[17,'nk'],[0,'hd'],[12,'rhp'],[11,'lhp']]:
                    pointDict = {}
                    kpt = datum.poseKeypoints[index][p[0]]
                    if kpt[2] > 0 :
                        kptName = p[1]
                        point = kpt[:2].tolist()
                        v = str(1)
                        kptLabelName = poseID+'_xx_'+kptName+'_'+v
                        
                        pointDict['label'] = kptLabelName
                        pointDict['fill_color'] = None
                        pointDict['line_color'] = None
                        pointDict['points'] = [point]
                        pointDict['shspe_type'] = 'point'
                        total_point.append(pointDict)
                        
        labelJson = {}
        labelJson['version'] = '3.16.7'
        labelJson['flag'] = None 
        labelJson['shapes'] = total_point
        labelJson['lineColor'] = [0,255,0,128]         
        labelJson['fillColor'] = [255,0,0,128] 
        labelJson['imagePath'] = relativePath
        labelJson['imageWidth'] = w
        labelJson['imageData'] = None 
        labelJson['imageHeight'] = h
        
               
    
        saveLabelmePath = os.path.join('.','annotations',fileName+'.json')
        with open(saveLabelmePath , 'w') as outfile:
            json.dump(labelJson, outfile)
                
                
        if visualize:
            cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo',(640,480))
            cv2.imshow('demo',datum.cvOutputData)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    run()

