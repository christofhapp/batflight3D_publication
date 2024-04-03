# What is it about?
The code in this repository is used to generate the result plots of the publication FUTURE_PUBLICATION. The publication describes a workflow to do a 3D calibration of two synchronized, stationary thermal cameras based on a GPS drone flight track. This can be used e.g. to track flying animals at night. The plots are based on the data in the data/ folder.

# Quick Start
install python with numpy, matplotlib, opencv-python, utm, pandas, scikit-image and scikit-learn

run **plot_statistics.py** to see the final 3D error plots of all 3D calibrations and all the 3D flights if you click a boxplot

run **animal_flightpath_example.py** to see how you get from the synchronized images to a final 3D flight path reconstruction by using the 3D calibration calculated before

![rotate360](https://github.com/christofhapp/batflight3d/assets/51400845/9cc5aaab-6d7e-4ffc-98c5-866720175726)


# Data
The data folder contains a folder for each drone flight filmed by the cameras. It contains the drone GPS track, an imagelist of the synchronized cameras, the result of 2D moving object detection of each camera separately and 5-10 manually selected drone points in the 2D images according to the publication.

```bash
data/
├── ...
├── Offenhausen_2022-06-22_21-53
│   ├── DRONE_PTS.csv
│   ├── imagelist.csv
│   ├── IM_PTS_cam1.csv
│   ├── IM_PTS_cam2.csv
│   ├── points_clicked.csv
├── ...
```

### DRONE_PTS.csv
```
datetime,lat,lon,height
2022-06-23 19:42:26.386,49.42217084716835,11.447858470221677,0.0
2022-06-23 19:42:26.486,49.42217088440474,11.447858501829796,0.0
2022-06-23 19:42:26.586,49.42217094773711,11.447858540094895,0.0
...
```
This file contains the GPS data from the drone

### imagelist.csv

This file contains the image number for the synchronized cameras and a timestamp. This file is important to know when the image was taken in order to synchronize it to the drone GPS points.

```
nr,datetime
1, 2022-06-23 21:41:58.701641
3, 2022-06-23 21:42:02.034981
5, 2022-06-23 21:42:05.368321
...
```

### IM_PTS_cam1.csv

This file contains all possible x,y points of the drone in the images detected with 2D image processing methods. We provide this file instead of the raw images, because they would consume lots of gigabytes of data.
The file (for camera 1 in this case) contains addional information like area, max_scoreImg, max_innerToRing which are not discussed in detail, but they help to make a final decision of a corresponding point pair.
max_scoreImg: A 2D score image is thresholded, but the maximum is saved for later use
max_innerToRing: The max of the score image is compared to it's surrounding (also for later use)

```
x,y,area,max_scoreImg,max_innerToRing,nr
405.1,457.19,52.0,22.0,10.9,10
58.06,326.57,56.0,22.0,15.2,12
...
```

### points_clicked.csv
This file contains 5-10 manually clicked drone points to do a preliminary 3D calibration and synchronization between the cameras and the drone. This is also provided instead of the whole image dataset.
x1 and y1 belong to camera 1. The number again corresponds to the image number of the cameras (which is the same for both cameras as they are synchronized).

```
nr,datetime,x1,y1,x2,y2
4357,2021-08-11 13:55:15.399708800,433.95,359.41,387.04,357.7
6159,2021-08-11 13:56:15.466495600,354.0,278.4,292.0,277.71
7141,2021-08-11 13:56:48.199894400,447.18,73.15,382.65,83.99
...
```

### Folder bat_recording
There is one special folder **bat_recording** in the folder **Offenhausen_2022-06-23_21-41/images**. It contains a cut out sequence of a measured animal flight path at 10:56 p.m. including the **imagelist.csv** and two folders containing all the images.
In the code section you can find a script for **animal_flightpath_example.py** to calculate the final animal flightpath from this data source.

Open the gif **zoomed_recording.gif** in the folder **Offenhausen_2022-06-23_21-41/images/bat_recording** to see a zoomed in video of that animal flying close to the wind turbine:

<img src="https://github.com/christofhapp/batflight3d/blob/main/data/Offenhausen_2022-06-23_21-41/images/bat_recording/zoomed_recording.gif" width="50%">



# Code

**generate_DRONE_SIM_PTS.py**

At first, we calculate a CSV **DRONE_SIM_PTS.csv** with the python code **generate_DRONE_SIM_PTS.py** for each folder which contains the synchronization between the cameras and the drone and links the 2D points of both cameras to the corresponding 3D drone GPS points.
```
nr,datetime,x1,y1,x2,y2,lat,lon,height
2671,2021-08-11 13:54:19.199596399,14.3,285.14,0.0,273.5,48.66520753323397,9.836363320580258,18.4
2672,2021-08-11 13:54:19.232929800,15.07,285.15,0.13,272.93,48.66520753323397,9.836363320580258,18.4
2673,2021-08-11 13:54:19.266263200,15.94,285.0,0.5,272.76,48.66520746098632,9.836363289842678,18.6
```
If you execute the script, you get asked to type in a number from *1*-*16* or *all*. If you type in a number, the DRONE_SIM_PTS of the folder get calculated and you additionally see the synchronization plots. If you type all, the DRONE_SIM_PTS of all folders get calculated. That needs some time and no synchronization plots are shown. You don't have to execute this script as all the files are already pre-calculated for you to run the next file **plot_statistics.py**.
But it can be used for a new dataset of course or as a proof how this data was generated.

### plot_statistics.py

long:
Then, by executing **plot_statistics.py**, we take the **DRONE_SIM_PTS.csv** of all folders, do all the 3D calibrations, calculate 3D errors via cross validation and show them in a boxplot. Furthermore we calculate the 3D errors in dependency of their distance to the camera system. The boxplots can be clicked to show the 3D flightpaths for 3D calibration, 3D reconstruction from the images and cross-validated evaluation as well as a speedplot of the drone based on the GPS data compared to one based on the image reconstructed 3D flight paths.

short: 
just execute the script.

### animal_flightpath_example.py
This script calculates an animal flightpath measured after the calibration flight in Offenhausen from a set of synchronized images in the folders cam1 and cam2 and the **imagelist.csv** using the calibration already achieved by the **generate_DRONE_SIM_PTS.py**. Just run the script and it will put **IM_PTS_cam1.csv** and **IM_PTS_cam2.csv** in the folder and produce plots characterizing the measured flight path.

### batflight3D.py
The module containing all the collected functions.

### batflight3dGUI.py
The module containing all the collected GUI functions
