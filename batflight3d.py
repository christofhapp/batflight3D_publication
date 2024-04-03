import json
import numpy as np
import os
import re
import time
import pandas as pd
import cv2
import utm

# for image detection
from skimage.measure import label, regionprops
from sklearn import cluster

config = {}
Options = {}
def load_config_json(config_filename):
    global config
    if config_filename != '':
        f = open(config_filename, 'r')
        config = json.load(f)
        if 'ref' in config:
            convert_ref_latlon_to_gps_offset_in_meters()

# DRONE_SIM_PTS: Dataframe: 'nr', 'datetime', 'x1', 'y1', 'x2', 'y2', 'lat', 'lon', 'height','objX','objY'
def load_DRONE_SIM_PTS(ppath):
    DRONE_SIM_PTS = pd.read_csv(ppath)[['nr', 'datetime', 'x1', 'y1', 'x2', 'y2', 'lat', 'lon', 'height']]
    DRONE_SIM_PTS.datetime = DRONE_SIM_PTS.datetime.astype(dtype='datetime64[ns]')
    DRONE_SIM_PTS['time'] = (DRONE_SIM_PTS.datetime - DRONE_SIM_PTS.datetime[0]).to_numpy().astype('float') / 1e9  # in s since start

    # calculate objX, objY
    gpsExtended = convert_latlon_to_utm(DRONE_SIM_PTS)
    convert_ref_latlon_to_gps_offset_in_meters()
    DRONE_SIM_PTS['objX'] = gpsExtended['utm_X'] - gps_x_offset
    DRONE_SIM_PTS['objY'] = gpsExtended['utm_Y'] - gps_y_offset

    return DRONE_SIM_PTS

class Calibration3D:

    def __init__(self):
        self.filename = ''

    def calc(self,imPTS,objPTS,pnpflag):

        self.imPTS = imPTS
        self.objPTS = objPTS
        self.pnpflag = pnpflag

        if isinstance(imPTS, pd.DataFrame):
            imPTS_L = imPTS[['x1', 'y1']].copy().to_numpy().astype('float32')
            imPTS_R = imPTS[['x2', 'y2']].copy().to_numpy().astype('float32')
        else:
            imPTS_L = imPTS[:,:2]
            imPTS_R = imPTS[:,2:]

        if isinstance(objPTS, pd.DataFrame):
            objPTS = objPTS[['objX', 'objY', 'height']].copy().to_numpy().astype('float32')

        camintrinsicMatrix1 = np.array(config['camintrinsicMatrix1'], dtype='float32')
        camintrinsicMatrix2 = np.array(config['camintrinsicMatrix2'], dtype='float32')
        distortion1 = np.array(config['distortion1'], dtype='float32')
        distortion2 = np.array(config['distortion2'], dtype='float32')

        retval, rvec1, self.tvec1 = cv2.solvePnP(objPTS, imPTS_L, camintrinsicMatrix1, distortion1, flags=pnpflag)
        retval, rvec2, self.tvec2 = cv2.solvePnP(objPTS, imPTS_R, camintrinsicMatrix2, distortion2, flags=pnpflag)

        if not isinstance(rvec1,np.ndarray) or not isinstance(rvec2,np.ndarray): # if no calibration can be achieved (happens sometimes at time synchronization function)
            return 0

        R1 = cv2.Rodrigues(rvec1)[0] # rotation-vector to matrix
        R2 = cv2.Rodrigues(rvec2)[0]


        self.Pr1 = camintrinsicMatrix1 @ np.hstack((R1, self.tvec1))
        self.Pr2 = camintrinsicMatrix2 @ np.hstack((R2, self.tvec2))

        self.Pos1 = (-R1.transpose() @ self.tvec1).flatten()
        self.Pos2 = (-R2.transpose() @ self.tvec2).flatten()
        Dir1 = R1.transpose() @ [0, 0, 1]
        Dir2 = R2.transpose() @ [0, 0, 1]

        self.Dir1 = Dir1 / np.linalg.norm(Dir1)
        self.Dir2 = Dir2 / np.linalg.norm(Dir2)

        self.R1 = R1
        self.R2 = R2

        self.R = R2 @ np.linalg.inv(R1)
        T = (R2 @ (self.Pos1-self.Pos2)).flatten()
        t_x = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
        self.T = T
        self.E = t_x @ self.R
        self.F = np.linalg.inv(camintrinsicMatrix2).transpose() @ self.E @ np.linalg.inv(camintrinsicMatrix1)

        # latlon of cameras
        self.Pos1_latlon = convert_utm_to_latlon(self.Pos1)
        self.Pos2_latlon = convert_utm_to_latlon(self.Pos2)

        # calibration error (biased)
        self.reconPTS = reconstruct3D(self, self.imPTS)
        self.errorvektor = self.reconPTS - objPTS  # reconPTS - objPTS
        self.error = np.linalg.norm(self.errorvektor, axis=1)

    def printCalibrationInfo(self):
        print('### CALIBRATION -- with', len(self.objPTS), 'Points')
        print('distance cameras: ',round(np.linalg.norm(self.Pos1-self.Pos2),2),'m')
        print('elevation angle camera 1: ', round(np.arcsin(self.Dir1[2]) * 180 / np.pi,2),'°')
        print('elevation angle camera 2: ', round(np.arcsin(self.Dir2[2]) * 180 / np.pi,2),'°')
        print('azimut angle camera 1: ', round(vectorToGeoAngle(self.Dir1),2),'°')
        print('azimut angle camera 2: ', round(vectorToGeoAngle(self.Dir2),2),'°')
        print()
        print('Calculated Camera Positions (lat,lon):')
        print('Camera 1: https://www.google.com/maps/place/'+str(self.Pos1_latlon[0])+','+str(self.Pos1_latlon[1]))
        print('Camera 2: https://www.google.com/maps/place/'+str(self.Pos2_latlon[0])+','+str(self.Pos2_latlon[1]))
        print()
        print('(self-biased) Calibration Errors:')
        print('25\% quantile = ' + str(round(np.quantile(self.error, .25), 2)))
        print('Median = ' + str(round(np.quantile(self.error, .5), 2)))
        print('75\% quantile = ' + str(round(np.quantile(self.error, .75), 2)))


def reconstruct3D(calib3D,imPTS):

    imPTS_L = imPTS[['x1','y1']].to_numpy()
    imPTS_R = imPTS[['x2','y2']].to_numpy()

    # DLT Method from Multiple View Geometry
    reconPTS = cv2.triangulatePoints(calib3D.Pr1,calib3D.Pr2,imPTS_L.transpose(),imPTS_R.transpose())

    # homogeneous coordinates to cartesian
    reconPTS /= reconPTS[3]
    reconPTS = reconPTS[0:-1].transpose()
    return reconPTS

def vectorToGeoAngle(vec):
    euler = np.arctan2(vec[1], vec[0]) * 180 / np.pi
    geo = 90-euler
    if geo<0:
        return geo+360
    else:
        return geo


def convert_latlon_to_utm(df):
    df[['utm_X', 'utm_Y','utm_zone','utm_north']] = df[['lat', 'lon']].apply(lambda x: utm.from_latlon(x.iloc[0], x.iloc[1]), axis=1, result_type='expand')
    return df

# xyz: numpy-array [[x1,y1,z1],[x2,y2,z2],...]
def convert_utm_to_latlon(xyz):
    # make a matrix if it is a vector
    xyz = np.reshape(xyz,(-1,3))
    utmx, utmy, zone, north = utm.from_latlon(config['ref'][0], config['ref'][1])
    lat,lon = utm.to_latlon(xyz[:,0] + gps_x_offset, xyz[:,1] + gps_y_offset, zone, north)
    if len(lat)==1:
        lat = lat[0]
        lon = lon[0]
    return lat,lon

def convert_ref_latlon_to_gps_offset_in_meters():
    global gps_x_offset, gps_y_offset
    df_gps_offset = convert_latlon_to_utm(pd.DataFrame({'lat':[config['ref'][0]], 'lon':[config['ref'][1]]}))
    gps_x_offset = df_gps_offset['utm_X'].to_numpy()[0]
    gps_y_offset = df_gps_offset['utm_Y'].to_numpy()[0]


def load_CSV_Imagelist_path(lpath):
    global DD,L,R

    DD = pd.read_csv(lpath)
    if 'datetime' in DD.keys():
        DD.datetime = DD.datetime.astype(dtype='datetime64[ns]')
        DD['time'] = (DD.datetime - DD.datetime[0]).to_numpy().astype('float') / 1e9  # in s since start

    L = np.arange(1, len(DD) + 1)
    R = L

def loadDronePoints(path):
    global objPTS, objPTS_filename, objPTS_time, objPTS_latlon, objPTS_UTM
    gpsRaw = pd.read_csv(path)

    # extend GPS DataFrame with utm_X, utm_Y and altitude (from lat,lon,height)
    gpsExtended = convert_latlon_to_utm(gpsRaw)
    objPTS_UTM = gpsExtended[['utm_X','utm_Y','height']].to_numpy()
    convert_ref_latlon_to_gps_offset_in_meters()
    objPTS = objPTS_UTM.copy()
    objPTS[:, 0] = objPTS_UTM[:, 0] - gps_x_offset
    objPTS[:, 1] = objPTS_UTM[:, 1] - gps_y_offset

    # global objPTS used in other functions
    objPTS_filename = path
    objPTS_latlon = gpsRaw[['lat', 'lon']].to_numpy()
    objPTS_time = gpsRaw['datetime'] # str timestamp
    objPTS_time = objPTS_time.astype(dtype='datetime64[ns]') # to pd.timestamp
    objPTS_time = objPTS_time - objPTS_time[0] # timedelta since start
    objPTS_time = objPTS_time.to_numpy() # to numpy (ns)
    objPTS_time = objPTS_time.astype('float') # to float (ns)
    objPTS_time = objPTS_time/1e9 # time in s since start
    #gpsExtended = convert_height_to_altitude(gpsExtended)

def add_time(imPTS):
    imPTS['time'] = (imPTS.datetime - imPTS.datetime[0]).to_numpy().astype('float') / 1e9  # in s since start
    return imPTS

def load_STEREO_IM_PTS(points_path):
    imPTS = pd.read_csv(points_path)
    imPTS_filename = os.path.basename(points_path)

    imPTS.datetime = imPTS.datetime.astype(dtype='datetime64[ns]')
    imPTS = add_time(imPTS)

    return imPTS,imPTS_filename

def estimateSynchronizationCamToDrone(PTS,search='global'):
    global dt_cam_drone, dt_cam_drone_filename

    meandistance = []

    # 4-6 points
    if search=='global':
        imPTS = PTS.sort_values(by=['nr'], ignore_index=True)
        print(len(imPTS),' STEREO_IM_PTS found for synchronization')
        DDimgPTS = pd.merge(DD, imPTS, on='nr', how='inner', suffixes=(None, '_right')).sort_values(by=['nr'])
        timediff_img = (DDimgPTS.time - DDimgPTS.time.iloc[0]).to_numpy()[1:]
        ix_vec = [0]
        for j in range(0, len(DDimgPTS)-1):
            ix_vec.append(np.argmin(np.abs(objPTS_time-objPTS_time[0]-timediff_img[j]))) # Find objPTS Index mit selbem Zeitabstand wie drüber, zumindest das Näheste
        stop_index = len(objPTS)-1 # -1 to be exact, but sometimes on the end of a DJI file you have more than one

    # over 10 points
    elif search=='local':
        DRONE_SIM_PTS = PTS
        print('dt_cam_drone and ',len(DRONE_SIM_PTS),' DRONE_SIM_PTS found. Doing a local search around "dt_cam_drone".')
        DDimgPTS = DRONE_SIM_PTS
        timediff_img = (DDimgPTS.time - DDimgPTS.time.iloc[0]).to_numpy()[1:]
        search_width = 20
        ix_vec = [] # vector of objPTS with same time distances than image points
        ix_vec.append(np.argmin(abs(objPTS_time - DDimgPTS.time.iloc[0] + dt_cam_drone)) - search_width)
        for j in range(1,len(DDimgPTS)):
            ix_vec.append( np.argmin(np.abs(objPTS_time-objPTS_time[ix_vec[0]]-timediff_img[j-1])) )
        stop_index = ix_vec[-1] + 2 * search_width
    else:
        return 0

    ix_vec = np.array(ix_vec)

    while ix_vec[-1] < stop_index:
        objPTS4 = objPTS[ix_vec]
        meandistance.append([ix_vec.copy(), 1000]) # 1000 -> irgendein hoher Wert
        kCalib = Calibration3D()
        try:
            ret = kCalib.calc(DDimgPTS, objPTS4, cv2.SOLVEPNP_SQPNP)
        except:
            ret = 0
            print('\rCalibration - Error',end='')

        # ix_vec neu -> Finde objPTS Index der gemäß Bild-Zeit-Abstand am Nächsten ist
        print('\rIteration Drone Signal: ', ix_vec,end='')
        #print(ix_vec)
        ix_vec[0]+=1
        for j in range(1,len(DDimgPTS)):
            ix_vec[j] = np.argmin(np.abs(objPTS_time-objPTS_time[ix_vec[0]]-timediff_img[j-1]))

        if ret == 0:
            # print('Kalibrierungsfehler in rvec tvec')
            continue
        reconPTS = reconstruct3D(kCalib, DDimgPTS)
        distances = np.linalg.norm(reconPTS - objPTS4, axis=1)
        meandistance[-1][1] = distances.mean() # if calib worked: add real meandistance on pos 1

    meandistances_meters = [x[1] for x in meandistance]
    meandistance_ixvec = [x[0] for x in meandistance]

    ix_min_meandistance = np.argmin(meandistances_meters)
    ix_vec0 = meandistance_ixvec[ix_min_meandistance][0]
    dt_cam_drone = DDimgPTS.time.iloc[0]-objPTS_time[ix_vec0]

    print()
    print('Automatic estimation done! dt_cam_drone = '+str(round(dt_cam_drone,3))+' s | mean error between object points and reconstructed points = '+str(round(np.min(meandistances_meters),2))+' m')

    # Calculate list of len(L) with corresponding drone points
    DDdronePoints = correlateDronePointToImagelist(L, objPTS_time, dt_cam_drone)



    return dt_cam_drone, DDdronePoints, meandistances_meters

def correlateDronePointToImagelist(L,objPTS_time,dt_cam_drone):

    start=0
    DDdronePoints = np.zeros(len(L), dtype=int)
    for ii,ele in enumerate(L):
        if ii % 10000 == 0:
            print(ii, ' of ', len(L), ' images')
        for jj in range(start, len(objPTS_time)):
            if ele != 0:  # kein dummy
                deltat = objPTS_time[jj] - (DD.time[ii] - dt_cam_drone)
                if deltat >= 0.05: # t in s
                    break
                if abs(deltat)<0.05:
                    DDdronePoints[ii] = int(jj)
                    # Start inner Loop (dronetimes) again from current drone Index (and not from the beginning)
                    start = jj
                    break
    # Momentan werden ca 3 Bildern jeweils derselbe GPS-Punkt zugewiesen
    print(sum(DDdronePoints != 0), 'of ', len(L),' images successfully correlated to one of the ', len(objPTS_time), 'drone points')
    return DDdronePoints


def correlateDronePointsToSegmentationPoints(objPTS, imPTS, DDdronePoints, calib=0, corrThreshold=0):
    # DDdronePoints same length as LR and contains indices of drone point or 0

    DD['index'] = DD.index
    DDimPTS = pd.merge(DD, imPTS, on='nr', how='inner', suffixes=(None,'_right'))
    objPTSix = DDdronePoints[DDimPTS['index']]

    if len(objPTSix<10):
        print(objPTSix)
    zero_index_bool = objPTSix == 0
    print('Nr. of eliminated imagepoints, because no dronepoint could be found = ',sum(zero_index_bool))
    objPTScorr = objPTS[objPTSix[np.invert(zero_index_bool)]]
    objPTScorr = np.array(objPTScorr, dtype='float32')
    objPTScorr_time = objPTS_time[objPTSix[np.invert(zero_index_bool)]]
    objPTScorr_time = np.array(objPTScorr_time, dtype='float32')
    objPTScorr_latlon = objPTS_latlon[objPTSix[np.invert(zero_index_bool)]]
    objPTScorr_latlon = np.array(objPTScorr_latlon, dtype='float64')
    corrPTS = DDimPTS[np.invert(zero_index_bool)].reset_index(drop=True) # reset Index to make it the same as in objPTScorr (which is not a DataFrame)
    corrPTS['objX'] = objPTScorr[:, 0]
    corrPTS['objY'] = objPTScorr[:, 1]
    corrPTS['height'] = objPTScorr[:, 2]
    corrPTS['lat'] = objPTScorr_latlon[:, 0]
    corrPTS['lon'] = objPTScorr_latlon[:, 1]
    corrPTS['objTime'] = objPTScorr_time

    # are the reconstructed points close enough to the drone to be the drone or maybe something else?
    if corrThreshold > 0:
        if calib != 0:
            reconPTS = reconstruct3D(calib, corrPTS) # calculate 3D Reconstruction
            print('Length points before reduction based on calibration = ',len(corrPTS))
            corrPTS.loc[:, 'droneDistance'] = np.linalg.norm(reconPTS - objPTScorr, axis=1) # add droneDistance as decision criterion
            #corrPTS = corrPTS.loc[corrPTS['droneDistance'] <= corrPTS['droneDistance'].median() * 10] # allow just imPTS that are 10 times the median away from drone
            corrPTS = corrPTS.loc[corrPTS['droneDistance'] <= corrThreshold]  # allow just imPTS that are 15 meters from drone
            #print('10 times median = ', corrPTS['droneDistance'].median() * 10, ' m')
            print('Length points after reduction based on ',corrThreshold,' m fence = ', len(corrPTS))
            corrPTS['min'] = corrPTS.groupby('nr')['droneDistance'].transform(lambda x: x.min()) # find minima if there are more than one point per image
            corrPTS = corrPTS.loc[corrPTS['droneDistance'] == corrPTS['min']] # allow just the minimum distance if there are two points still present
            print('Length points after reduction based on calibration = ',len(corrPTS))
        else:
            print('no Calibration found for calculating ')

    return corrPTS

def doCalibration(DRONE_SIM_PTS):
    global calib
    print('do Calibration')

    imPTS = DRONE_SIM_PTS[['x1','y1','x2','y2']]
    objPTS = DRONE_SIM_PTS[['objX','objY','height']]

    calib = Calibration3D()
    calib.calc(imPTS, objPTS, cv2.SOLVEPNP_SQPNP)
    calib.printCalibrationInfo()

    print('calc. Calibration (' + str(len(DRONE_SIM_PTS)) + ' Points)')
    return calib

def load_and_correspond_PTS(calib,xyL_all,xyR_all):

    imPTS_list = []
    for idx, ele in enumerate(L):
        if L[idx] != 0 and R[idx] != 0: # not dummy
            xyL = xyL_all.loc[xyL_all['nr'] == DD.nr[idx]][['x','y','area','max_scoreImg','max_innerToRing']].to_numpy() # old files: x,y
            xyR = xyR_all.loc[xyR_all['nr'] == DD.nr[idx]][['x','y','area','max_scoreImg','max_innerToRing']].to_numpy()
            imPTS_new, epilinesR = detectCorrespondingPoints(calib, xyL, xyR, idx)
            if len(imPTS_new) > 0:
                imPTS_list = imPTS_list+imPTS_new
            print('\rImage '+str(idx)+' of '+str(len(L)),end='')

    print('')
    print('len ImagePTS_list: ',len(imPTS_list))
    imPTS = pd.DataFrame(imPTS_list, columns=['x1', 'y1', 'x2', 'y2', 'nr','datetime'])
    imPTS = add_time(imPTS)
    return imPTS

def detectCorrespondingPoints(calib,xyL,xyR, i): # EPILINES

    # Thresholding Score Image, result: uncorresponding points (candidates) in both images
    # this function: make correspondence over epipolar geometry
    # calculate epipolar line in image of cam2 for point in image of cam1 and calculate corresponding points

    imPTS = []
    epiLinesR = []
    candidates = []

    #print(' | Anzahl unkorrespondierender Pkte L=',len(xyL),' R=',len(xyR),end='')

    for i_L,ptL in enumerate(xyL):
        # Calculate Epiline for Point
        epiline = calib.F @ [ptL[0], ptL[1], 1]  # F dot [x,y,1]  @ = dotProduct
        d = - epiline[2] / epiline[1]  # like k*x+d
        k = -epiline[0] / epiline[1]
        L1 = np.array([0, d, 0])  # vektor (0,d)
        L2 = np.array([1, d + k, 0])  # vektor (1,k)

        for i_R,ptR in enumerate(xyR):
            pt = ptR[:2].astype('float') # because if point has other values than float the whole np array gets casted to object in some cases
            distFromEpiline = np.linalg.norm(np.cross(np.append(pt, 0) - L1, L2 - L1)) / np.linalg.norm(L2 - L1)  # |(point - point_line) x direction_line| / | direction_line |
            areaRatio =  ptL[2]/ptR[2]

            # accept point pair if R close enough to epi-polar line and if area ratio is close enough
            if distFromEpiline <= config['maxDistFromEpiline'] and areaRatio > 1/config['areaRatio'] and areaRatio < config['areaRatio']:
                candidates.append([i_L,i_R,distFromEpiline,k,d])
                #print('ptRx=',ptR[1],'ptRy=',ptR[0],' areaRatio: ',round(areaRatio,2), 'dist: ',round(dist,2))

    candidates = np.array(candidates)

    # find more points on the same epi line
    if len(candidates)>=2:
        # exclude points that got matched to the epiline accidentally

        # for 0 column:
        unL, indL, countL = np.unique(candidates[:,0], return_index=True, return_counts=True)
        candidates = candidates[indL][countL==1]
        # for 1 column:
        unR, indR, countR = np.unique(candidates[:,1], return_index=True, return_counts=True)
        candidates = candidates[indR][countR==1]

        if (countL>1).any() or (countR>1).any():
            # countL countR points got excluded
            pass

    # number of corresponding points: len(candidates)

    # Add resulting correspondences
    for candidate in candidates:
        i_L = int(candidate[0])
        i_R = int(candidate[1])
        imPTS.append([ xyL[i_L][0], xyL[i_L][1], xyR[i_R][0], xyR[i_R][1] ])
        epiLinesR.append([candidate[3], candidate[4]])

    # imPTS = [[imgL_x, imgL_y, imgR_x, imgR_y],...]
    # epiLines = [[k,d],...]
    #imPTS = np.array(imPTS)
    epiLinesR = np.array(epiLinesR)

    # "ix" is the global Listenindex for the LR list
    if len(imPTS)>0:
        #df = pd.DataFrame(imPTS, columns=['x1', 'y1', 'x2', 'y2'])
        if L[i] != 0 and R[i] != 0:  # no dummy
            for elem in imPTS:
                elem.append(DD.nr[i])
                elem.append(DD.datetime[i])
        else:
            print('L[i] or R[i] == none!')

    return imPTS, epiLinesR

def calculateCorrThreshold(error):
    # like whisker calculation with whis=10
    Q1 = np.quantile(error, .25)
    Q3 = np.quantile(error, 0.75)
    whis=10
    return Q3+whis*(Q3-Q1)

def calcSpeed(t,xyz,z):
    t = t - t.min()
    timediffs = []
    placediffs = []
    # z .. take mean of z frames difference (smoothing factor)
    for i in range(len(xyz) - z):
        timediffs.append(t[i + z] - t[i])
        placediffs.append(np.linalg.norm(xyz[i + z] - xyz[i]))
    ztimes = t[z:]
    #distance = np.linalg.norm(xyz - (calib.Pos1 + (calib.Pos2 - calib.Pos1) * 0.5), axis=1)
    timediffs = np.array(timediffs)
    placediffs = np.array(placediffs)
    speed_abs_kmh = placediffs / timediffs * 3.6
    return ztimes,speed_abs_kmh


class Images:
    def __init__(self):
        self.pathL = ""
        self.pathR = ""
        self.filenamesL = []
        self.filenamesR = []
        self.i_begin_file = 0
        self.i_end_file = 0

        # for detection
        self.stackL = []
        self.stackR = []
        self.i_previous=-1

    def chooseImagesInFolder(self, pathL, pathR):
        global Options

        self.pathL = pathL
        self.pathR = pathR

        self.filenamesL = os.listdir(pathL)
        self.filenamesL.sort(key=lambda f: int(re.sub('\D', '', f))) # \D are all NON-Digits re.sub:
        self.filenamesR = os.listdir(pathR)
        self.filenamesR.sort(key=lambda f: int(re.sub('\D', '', f)))

        self.i_begin_file = 0
        self.i_end_file = len(self.filenamesL)-1

        self.loadImagesInFolder(self.i_begin_file)

        pic0 = self.pics['L'][0]
        print('## Image ##')
        print('im.shape = ',pic0.shape)
        Options['image_shape'] = pic0.shape

        self.backgroundL = np.zeros(Options['image_shape'], np.uint8)
        self.backgroundR = np.zeros(Options['image_shape'], np.uint8)

        Options['clim_original'] = [pic0.min(),pic0.max()]
        print('min_value = ',pic0.min())
        print('max_value = ', pic0.max())


    def loadImagesInFolder(self,i):


        if i>config['bg_stacksize']:
            if i > config['bg_stacksize']:
                self.firstImgIdxOfImgFileL = i + 1 - config['bg_stacksize'] + 1
                self.firstImgIdxOfImgFileR = i + 1 - config['bg_stacksize'] + 1

            # Append just one Image per Step (going forward +1)
            if self.i_previous>=0 and i - self.i_previous == 1 and len(self.pics['L']) >= config['bg_stacksize']:
                    self.pics['L'].append(self.openImage(self.pathL + '/' + self.filenamesL[i]))
                    self.pics['R'].append(self.openImage(self.pathR + '/' + self.filenamesR[i]))
                    self.pics['L'].pop(0)
                    self.pics['R'].pop(0)
            # Append whole Stack at once (jumping to Image | going backwards)
            else:
                self.pics = {'L': [], 'R': []}
                for j in range(config['bg_stacksize']-1,-1,-1):
                    self.pics['L'].append(self.openImage(self.pathL + '/' + self.filenamesL[i-j]))
                    self.pics['R'].append(self.openImage(self.pathR + '/' + self.filenamesR[i-j]))
        # Begin of Images < Stacksize
        else:
            self.pics = {'L': [], 'R': []}
            self.pics['L'].append(self.openImage(self.pathL + '/' + self.filenamesL[i]))
            self.pics['R'].append(self.openImage(self.pathR + '/' + self.filenamesR[i]))

    def openImage(self,path):
        img = cv2.imread(path, -1)
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def detectPointsLoopPNG(self,path_c1,path_c2,step=1):

        nr = DD.nr.to_numpy()

        file = open(path_c1, "w")
        file.write("x,y,area,max_scoreImg,max_innerToRing,nr\n")
        file.close()
        file = open(path_c2, "w")
        file.write("x,y,area,max_scoreImg,max_innerToRing,nr\n")
        file.close()

        # Iterate over Images
        tall = time.time()
        for i in range(self.i_begin_file,self.i_end_file+1,step):
            status = 'Image i='+str(i)+' of '+str(self.i_end_file)
            print(status+'\r',end='')

            # Load Images into pics['L'] pics['R']
            t = time.perf_counter()
            self.loadImagesInFolder(i)
            print('\r',status,' --- calculation time: loadImagesInFolder: ', round((time.perf_counter() - t)*1e3), ' ms, ',end='')

            # Detect Points
            if len(self.pics['L'])>=config['bg_stacksize']:
                t = time.perf_counter()
                backgroundL, backgroundR, scoreImgL, scoreImgR, binaryL, binaryR, xyL, xyR = self.detection(i)
                print('detection(i): ', round((time.perf_counter() - t)*1e3), ' ms, ',end='')

                # Save PTS in Files xyL,xyR
                t = time.perf_counter()
                if isinstance(xyL,np.ndarray)  and len(xyL)>0:
                    file = open(path_c1, "a")
                    for row in xyL:
                        for elem in row:
                            file.write(str(elem)+',')
                        file.write(str(nr[i])+'\n')
                    file.close()

                if isinstance(xyR,np.ndarray)  and len(xyR)>0:
                    file = open(path_c2, "a")
                    for row in xyR:
                        for elem in row:
                            file.write(str(elem)+',')
                        file.write(str(nr[i])+'\n')
                    file.close()

                print('\rsave to file: ', round((time.perf_counter() - t)*1e3), ' ms',end='')
        print('Overall Calculation Time:',round(time.time()-tall,1))


    def detection(self,i):

        #t = time.perf_counter()

        if self.i_previous>=0 and i-self.i_previous != 1:
            self.stackL = []
            self.stackR = []

        self.i_previous = i

        if i - config['bg_stacksize'] >= self.i_begin_file:
            for j in range(i - config['bg_stacksize'] + len(self.stackL) , i+1):

                if L[j] != 0: # kein dummy
                    imgL = self.pics['L'][L[j] - self.firstImgIdxOfImgFileL]
                    self.stackL,scoreImgL,self.backgroundL,binaryL,xyL= self.detectionSingle(imgL,self.stackL,self.backgroundL)
                else:
                    xyL = []
                    scoreImgL = None
                    binaryL = None

                if R[j] != 0: # kein dummy
                    imgR = self.pics['R'][R[j] - self.firstImgIdxOfImgFileR]
                    self.stackR, scoreImgR, self.backgroundR,binaryR,xyR = self.detectionSingle(imgR,self.stackR,self.backgroundR)
                else:
                    xyR = []
                    scoreImgR = None
                    binaryR = None

                #print('Time after detectionSingle on both sides: ', round((time.perf_counter() - t) * 1000,2), 'ms')

            return self.backgroundL,self.backgroundR,scoreImgL, scoreImgR,binaryL,binaryR,xyL,xyR
        else:
            return None, None, None, None, None, None, [], []

    def detectionSingle(self, img, stack, background):

        # t = time.perf_counter()

        img = img.astype('int32')
        stack.append(img)

        if len(stack) == config['bg_stacksize']:
            # print('detectionSingle: len(stack) == config['bg_stacksize']')
            background = sum(stack[0:config['bg_high']]) / len(stack[0:config['bg_high']])
            # print('Time after sum background: ', round((time.perf_counter() - t) * 1000,2), 'ms')
            return stack, None, background, None, None

        elif len(stack) > config['bg_stacksize']:
            # print('detectionSingle: len(stack) > config['bg_stacksize']')
            # Background = Average of Images 0-7 of 10
            background = background + (stack[config['bg_high']] - stack[0]) / config['bg_high']
            # background = background.astype('int32')

            stack.pop(0)

            # print('Time after quick background: ', round((time.perf_counter() - t) * 1000,2), 'ms')

            scoreImg = img - background

            # current image (Infratec Kameras) +1K entspricht +100 in img
            # devide by 10 (scoreDevider): 10=1K 0...Tmin  255...Tmin+25.5K
            scoreImg2 = scoreImg / config['scoreDevider']
            scoreImg2 = cv2.max(scoreImg2, 0)  # about 1 ms (instead of 2 ms for the np method: scoreImg2[scoreImg2 < 0] = 0)
            scoreImg2[scoreImg2 > 255] = 255
            scoreImg2 = scoreImg2.astype('uint8')

            ret, binary = cv2.threshold(scoreImg2, config['Threshold_scoreImage'], 255, cv2.THRESH_BINARY)  # 10 sehr sensibel, viel Wolken, aber auch ziemlich genau, 13: safe! dennoch einige FP

            # Dilation
            binary_dil1 = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
            binary_dil2 = cv2.dilate(binary_dil1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            # Labeling
            labeled_dil2 = label(binary_dil2)
            labeled_dil2 = labeled_dil2.astype('uint8')
            # Erosion
            labeled_erode1 = cv2.erode(labeled_dil2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            labeled_orig = cv2.erode(labeled_erode1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
            # Ring
            labeled_ring = cv2.subtract(labeled_dil2, labeled_erode1)
            labeled_ring = labeled_ring.astype('uint8')

            img_cv = ((img - img.min()) / config['scoreDevider'])

            props = regionprops(labeled_orig, scoreImg2)
            props_ring = regionprops(labeled_ring, img_cv)
            props_orig = regionprops(labeled_orig, img_cv)

            xy = []
            for i, prop in enumerate(props):
                # if prop.area >= 1:
                # euler_number: number of components minus holes (so if label has more than one component, it gets sorted out
                # Threshold_objWarmerThanRing: object has to be warmer than it's surrounding

                # Vorauswahl
                step1 = False
                if prop.euler_number == 1 and prop.area < config['area_max'] and props_orig[i].intensity_max > props_ring[i].intensity_max + config['Threshold_objWarmerThanRing']:
                    xy.append([round(prop.weighted_centroid[1], 2), round(prop.weighted_centroid[0], 2), prop.area, round(prop.intensity_max, 1), round(props_orig[i].intensity_max - props_ring[i].intensity_max, 1)])
                    step1 = True

                if step1 == False:
                    labeled_orig[labeled_orig == i + 1] = 0.5
                    labeled_ring[labeled_ring == i + 1] = 0.5

            xy = np.array(xy)

            #### check if Point belongs to a Cluster (reject if yes, e.g. clouds make clusters)
            if len(xy) > 0:
                # if at least 3 other points are within 25 px, then a cluster core point is found
                db = cluster.DBSCAN(eps=25, min_samples=3).fit(xy[:, 0:2])
                # add mean_intensity to DBSearch to detect a very warm object in front of a warm cluster?
                xy = xy[db.labels_ == -1]
            # print('Time after props: ', round((time.perf_counter() - t) * 1000,2), 'ms')

            return stack, scoreImg2, background, labeled_orig, xy

        return stack, None, None, None, None

def points2flightpaths(xyz, t, calib):
    # Kamera System
    camerasystem_position = calib.Pos1 + (calib.Pos2 - calib.Pos1) * 0.5
    camerasystem_direction = (calib.Dir1 + calib.Dir2) * 0.5

    # Filter points too far from camera system
    max_dist = 400  # 400
    distance_from_camerasystem = np.linalg.norm(xyz - camerasystem_position, axis=1)
    xyz = xyz[distance_from_camerasystem < max_dist]  # alle die weiter weg sind als 400 m raussschmeissen
    t = t[distance_from_camerasystem < max_dist]

    # Filter points behind camera system
    projection_vector = np.dot((xyz - camerasystem_position), camerasystem_direction)  # Punkte hinter Kamerasystem löschen!
    xyz = xyz[projection_vector > 0]
    t = t[projection_vector > 0]

    # True False vector with conditions
    onlyTrues = distance_from_camerasystem < max_dist
    onlyTrues[onlyTrues == True] = projection_vector > 0

    # Cluster if >= 3 pts within - 8 meters and 2 seconds
    time_dbscan = np.array([t / config['eps_sek'] * config['eps_m']]).transpose()  # Zeit normieren auf 8 Meter
    raumzeit_dbscan = np.hstack([xyz, time_dbscan])
    db = cluster.DBSCAN(eps=config['eps_m'], min_samples=config['min_samples']).fit(raumzeit_dbscan)
    # xy = xy[db.labels_ == -1]

    pts_flightpathnr = db.labels_
    # pts_flightpathnr[pts_flightpathnr==-1] = pts_flightpathnr.max()+1

    oT = pts_flightpathnr != -1
    pts_flightpathnr = pts_flightpathnr[oT]
    xyz = xyz[oT]
    t = t[oT]

    onlyTrues[onlyTrues == True] = oT

    return xyz, t, pts_flightpathnr, onlyTrues

def load_manual_GPS_coordinates_csv(points_path):

    manual_gps_coordinates = pd.read_csv(points_path)

    # lat,lon -> utm - gps_offset

    if 'x' in manual_gps_coordinates:
        df_gps_offset = convert_latlon_to_utm(manual_gps_coordinates)
        gps_x_offset0 = df_gps_offset['utm_X'].to_numpy()[0]
        gps_y_offset0 = df_gps_offset['utm_Y'].to_numpy()[0]
        df_XYZ = manual_gps_coordinates[['x', 'y']] + [gps_x_offset0, gps_y_offset0] - [gps_x_offset,gps_y_offset]
        df_XYZ['height'] = manual_gps_coordinates['height']
    else:
        df_XYZ = manual_gps_coordinates[['lat','lon']].apply(lambda x: utm.from_latlon(x[0], x[1]), axis=1, result_type='expand')[[0,1]]-[gps_x_offset, gps_y_offset]
        df_XYZ['height'] = manual_gps_coordinates['height']

    trackPTS = df_XYZ.to_numpy()

    return trackPTS