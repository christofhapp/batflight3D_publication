import batflight3dGUI
import batflight3d
import pandas as pd
import os


print('## Calculation of the corresponding Drone-Stereo-Image-Points ##')
print('Please choose which flight ID you want to calculate the DRONE_SIM_PTS for (1-16).')
print('This script will put the calculated DRONE_SIM_PTS.csv in the corresponding folder and output the synchronization plots.')
print('OR you can type "all" and it calculates all of them: that needs time and will not produce any plots, but puts all DRONE_SIM_PTS.csv in the corresponding foldes.')
option = input('type a flight ID from 1-16 or "all" (Enter): ')

dirs = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f))]

# Option 1: Calculate all without displaying anything
if option=='all':
    plot = False

# Option 2: Calculate only one folder and calculate
else:
    dirs = [dirs[int(option)-1]]
    plot = True

print('### Directories chosen:',dirs,'###')
print('After calculation has finished the DRONE_SIM_PTS.csv will be put in the above directories and can be used by the "plot_statistics.py"')

for folder in dirs:

    # load camera intrinsics, location info, ....
    if 'Turbine1' in folder:
        batflight3d.load_config_json(r'data\parameters_Turbine1.json')
    else:
        batflight3d.load_config_json(r'data\parameters_Turbine2.json')

    # load imagelist
    batflight3d.load_CSV_Imagelist_path('data\\'+folder+'\\imagelist.csv')

    # load Drone Points
    batflight3d.loadDronePoints('data\\'+folder+'\\DRONE_PTS.csv')

    # load clicked points (4-6 points)
    imPTS_prelim, imPTS_prelim_filename = batflight3d.load_STEREO_IM_PTS('data\\'+folder+'\\points_clicked.csv')

    print('\n### estimate synchronization of clicked points and the drone data AND calculate List of length(imagelist) containing corresponding drone points')
    dt_cam_drone_prelim, DDdronePoints_prelim, meandistances_meters_global = batflight3d.estimateSynchronizationCamToDrone(imPTS_prelim,search='global')
    print('dt_cam_drone_prelim: ',dt_cam_drone_prelim)

    print('\n### calculate Drone-Stereo-Image-Points of clicked points')
    print('if corrThreshold >=0, a global calib has to be defined in the module')
    DRONE_SIM_PTS_prelim = batflight3d.correlateDronePointsToSegmentationPoints(batflight3d.objPTS, imPTS_prelim, DDdronePoints_prelim, corrThreshold=0)

    print('\n### Preliminary Calibration')
    print('do a preliminary calibration (with the clicked points)')
    calib_prelim = batflight3d.doCalibration(DRONE_SIM_PTS_prelim)

    print('\n### load detected image points in both cameras')
    xyL_all = pd.read_csv('data\\'+folder+'\\IM_PTS_cam1.csv')
    xyR_all = pd.read_csv('data\\'+folder+'\\IM_PTS_cam2.csv')

    #batflight3d.config['maxDistFromEpiline'] = 6
    print('\n### calculate corresponding Image-Points in both cameras from the detected Points in each Camera')
    imPTS = batflight3d.load_and_correspond_PTS(calib_prelim,xyL_all,xyR_all)

    print('\n### calculate Drone-Stereo-Image-Points of all image points')
    print('corrThreshold is the Threshold which points get excluded, because they are not drone points according to the reconstruction with the preliminary calibration')
    DRONE_SIM_PTS_2 = batflight3d.correlateDronePointsToSegmentationPoints(batflight3d.objPTS, imPTS, DDdronePoints_prelim, corrThreshold=50, calib=calib_prelim)

    print('\n### also do a second calibration with the above gathered DRONE_SIM_PTS')
    calib2 = batflight3d.doCalibration(DRONE_SIM_PTS_2)

    print('\n### based on the calibration errors from calib2 calculate Threshold for 3D point Filtering')
    corrThres = batflight3d.calculateCorrThreshold(calib2.error)
    print('corrThreshold = ',corrThres)

    print('\n### update Drone-Stereo-Image-Points the better calibration')
    DRONE_SIM_PTS_3 = batflight3d.correlateDronePointsToSegmentationPoints(batflight3d.objPTS, imPTS, DDdronePoints_prelim, corrThreshold=corrThres, calib=calib2)

    print('\n### do a fine synchronization, local search')
    dt_cam_drone, DDdronePoints, meandistances_meters_local = batflight3d.estimateSynchronizationCamToDrone(DRONE_SIM_PTS_3,search='local')
    print('dt_cam_drone: ',dt_cam_drone)

    print('\n### do a third calibration with the excluded outliers')
    calib3 = batflight3d.doCalibration(DRONE_SIM_PTS_3)

    print('\n### based on the calibration errors from calib3 calculate Threshold for 3D point Filtering')
    corrThres = batflight3d.calculateCorrThreshold(calib3.error)
    print('corrThreshold = ',corrThres)

    print('\n### update Drone-Stereo-Image-Points after fine synchronization and with the better calibration')
    DRONE_SIM_PTS = batflight3d.correlateDronePointsToSegmentationPoints(batflight3d.objPTS, imPTS, DDdronePoints, corrThreshold=corrThres, calib=calib3)

    print('\n### save DRONE_SIM_PTS as calibration basis for future use')
    DRONE_SIM_PTS[['nr','datetime','x1','y1','x2','y2','lat','lon','height']].to_csv('data\\'+folder+'\\DRONE_SIM_PTS.csv')

    print('\n### calculate the final calibration (for showing information)')
    calib = batflight3d.doCalibration(DRONE_SIM_PTS)

    if plot:
        batflight3dGUI.synchronizationPlot(meandistances_meters_global,'global Synchronization (Flight ID '+option+')')
        batflight3dGUI.synchronizationPlot(meandistances_meters_local,'local fine Synchronization (Flight ID '+option+')',vline=20)
        batflight3dGUI.show()
