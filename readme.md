
# Pose Label Tool - labelme & OpenPose
## Installation
#### labelme
    $ pip3 install pyqt5
    $ pip3 install labelme==3.16.7 
#### Opnepose
    None
## Quick Start
#### Step1. Create annotations file by OpenPose 
1. put the image in the folder (./Labelme_Pose/images)
2. run python generator_labelme_json, and json file will be saved in the folder (./Labelme_Pose/annotations)

#### Step2. Modiy the json file which be created by Openpose
    $ cd ./Labelme_Pose/
    $ labelme
    (labelme UI) Click file -> Change Output Dir, modify path to ./Labelme_Pose/annotations
    (labelme UI) Click Open Dir, select images folder path (./Labelme_Pose/images)
    (Image will show the keypoints which is labeled by OpenPose)