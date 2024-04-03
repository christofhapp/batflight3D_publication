import batflight3d
import batflight3dGUI
import pandas as pd
import numpy as np

path = 'data\Turbine2_Setup4_2022-06-23_21-41\\'

# load camera intrinsics
batflight3d.load_config_json(r'data\parameters_Turbine2.json')

# load DRONE_SIM_PTS.csv of that location as a basis for the 3D calibration
DRONE_SIM_PTS = batflight3d.load_DRONE_SIM_PTS(path+'DRONE_SIM_PTS.csv')

# do the 3D calibration
calib = batflight3d.doCalibration(DRONE_SIM_PTS)
calib.printCalibrationInfo()

# load CSV imagelist
batflight3d.load_CSV_Imagelist_path(path+r'images\bat_recording\imagelist.csv')

# load Images
images = batflight3d.Images()
images.chooseImagesInFolder(path+r'images\bat_recording\cam1',path+r'images\bat_recording\cam2')

## detect points in PNGs
batflight3d.config['scoreDevider'] = 1
batflight3d.config['Threshold_scoreImage'] = 6
batflight3d.config['Threshold_objWarmerThanRing'] = 4
images.detectPointsLoopPNG(path+r'images\bat_recording\IM_PTS_cam1.csv',path+r'images\bat_recording\IM_PTS_cam2.csv')

print('\n### load detected image points in both cameras')
xyL_all = pd.read_csv(path + r'images\bat_recording\IM_PTS_cam1.csv')
xyR_all = pd.read_csv(path + r'images\bat_recording\IM_PTS_cam2.csv')

# batflight3d.config['maxDistFromEpiline'] = 6
print('\n### calculate corresponding Image-Points in both cameras from the detected Points in each Camera')
imPTS = batflight3d.load_and_correspond_PTS(calib, xyL_all, xyR_all)

print('\n### calculate reconstructed flying animal points')
reconPTS = batflight3d.reconstruct3D(calib, imPTS)

#batflight3d.config['eps_m']
#batflight3d.config['eps_sek']
#batflight3d.config['min_samples']
xyz,t,imPTS_flightpathnr,onlyTrues = batflight3d.points2flightpaths(reconPTS,imPTS['time'].to_numpy(),calib)

RC = batflight3dGUI.plot3D('')
RC.plotPointsColored(calib, reconPTS ,imPTS['time'],imPTS['nr'], imPTS_flightpathnr)

turbine2 = batflight3d.load_manual_GPS_coordinates_csv(path+r'images\GPS_WEA_Turbine2_XY.csv')
RC.plotPoints(calib, turbine2,0,0,color='black',size='5',marker='', ls='-')

print('Nr of detected flightpaths =',imPTS_flightpathnr.max())
xyz_track0 = xyz[imPTS_flightpathnr==0]
t_track0 = t[imPTS_flightpathnr==0]
ts, speed = batflight3d.calcSpeed(t_track0,xyz_track0,30)
speedplot = batflight3dGUI.plot(ts,speed,'tab:blue',2,'t / s','v / km/h')
heightplot = batflight3dGUI.plot(t_track0,xyz_track0[:,2],'tab:blue',2,'t / s','height / m')

print('')
print('Coordinates Animal 1:')
lat,lon = batflight3d.convert_utm_to_latlon(xyz_track0)
# numpy array
animal1_track = np.stack((lat,lon,xyz_track0[:,2]),axis=1)
# Dataframe
animal1_track_df = pd.DataFrame(animal1_track,columns=['lat','lon','height'])
print(animal1_track_df)


xyz_track1 = xyz[imPTS_flightpathnr==1]
t_track1 = t[imPTS_flightpathnr==1]
ts, speed = batflight3d.calcSpeed(t_track1,xyz_track1,30)
speedplot.plot(ts,speed,'tab:orange',2)
heightplot.plot(t_track1,xyz_track1[:,2],'tab:orange',2)



batflight3dGUI.show()

