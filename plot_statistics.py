import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import batflight3d
import batflight3dGUI


sets = [
    {
        'calibname': '2+3+4', # 1
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup1_2021-08-10_20-37\DRONE_SIM_PTS.csv'},
    {
        'calibname': '1+3+4',  # 2
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup1_2021-08-11_13-52\DRONE_SIM_PTS.csv'},
    {
        'calibname': '1+2+4',  # 3
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup1_2021-08-11_18-32\DRONE_SIM_PTS.csv'},
    {
        'calibname': '1+2+3',  # 4
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup1_2021-08-11_21-04\DRONE_SIM_PTS.csv'},
    {
        'calibname':'6+7+8', # 5
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup2_2023-09-27_13-56\DRONE_SIM_PTS.csv'},
    {
        'calibname': '5+7+8', # 6
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup2_2023-09-27_16-45\DRONE_SIM_PTS.csv'},
    {
        'calibname': '5+6+8', # 7
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup2_2023-09-27_17-10\DRONE_SIM_PTS.csv'},
    {
        'calibname': '5+6+7', # 8
        'DRONE_SIM_PTS_path': r'data\Turbine1_Setup2_2023-09-27_18-35\DRONE_SIM_PTS.csv'},
    {
        'calibname': '10', # 9
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup3_2022-06-22_21-53\DRONE_SIM_PTS.csv'},
    {
        'calibname': '9', # 10
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup3_2022-06-22_22-25\DRONE_SIM_PTS.csv'},
    {
        'calibname': '12', # 11
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup4_2022-06-23_21-29\DRONE_SIM_PTS.csv'},
    {
        'calibname': '11', # 12
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup4_2022-06-23_21-41\DRONE_SIM_PTS.csv'},
    {
        'calibname': '14', # 13
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup5_2022-06-25_21-03\DRONE_SIM_PTS.csv'},
    {
        'calibname': '13', # 14
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup5_2022-06-25_21-17\DRONE_SIM_PTS.csv'},
    {
        'calibname': '16', # 15
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup6_2022-06-26_20-46\DRONE_SIM_PTS.csv'},
    {
        'calibname': '15', # 16
        'DRONE_SIM_PTS_path': r'data\Turbine2_Setup6_2022-06-26_20-56\DRONE_SIM_PTS.csv'}
        ]

error = []
dist_from_camcenter = []

color=[]
errorvectors = []
DSP = []
means=[]
maxes=[]


batflight3d.load_config_json(r'data\parameters_Turbine1.json')


# do calibs of all flights
for i,set in enumerate(sets):
    print('load DRONE_SIM_PTS.csv Flight ID ', i, '...')
    if i<8:
        batflight3d.load_config_json(r'data\parameters_Turbine1.json')
    else:
        batflight3d.load_config_json(r'data\parameters_Turbine2.json')
    set['DSP'] = batflight3d.load_DRONE_SIM_PTS(set['DRONE_SIM_PTS_path'])
    # for later use
    set['config'] = batflight3d.config


def calibrate(DSP):
    calib = batflight3d.Calibration3D()
    ret = calib.calc(DSP[['x1', 'y1', 'x2', 'y2']], DSP[['objX', 'objY', 'height']], cv2.SOLVEPNP_SQPNP)
    return calib

def calcangles(R):

    # yaw = phi
    print('yaw:',np.arctan2(R[2, 1], R[2, 0]) * 180 / np.pi)
    # pitch = 90 - nu
    print('pitch:',np.arcsin(R[2, 2]) * 180 / np.pi)

    Rx = R.transpose()
    phi = np.arctan2(Rx[1,2],Rx[0,2])
    nu = np.arctan2(np.sqrt(Rx[0, 2]**2 + Rx[1, 2]**2), Rx[2, 2])
    psi = np.arctan2(Rx[2, 1], -Rx[2, 0])
    print('phi: ', phi*180/np.pi)
    print('nu: ', nu*180/np.pi)
    print('psi: ', psi * 180 / np.pi)


def changeRotationMatrixDegrees(R,deg_phi,deg_nu,deg_psi):
    # ZYZ Rotation

    R = R.transpose()

    # Euler
    rad_phi = deg_phi*np.pi/180
    rad_nu = deg_nu * np.pi / 180
    rad_psi = deg_psi * np.pi / 180

    phi=np.arctan2(R[1,2],R[0,2])+rad_phi
    nu = np.arctan2(np.sqrt(R[0, 2]**2 + R[1, 2]**2), R[2, 2])+ rad_nu
    psi = np.arctan2(R[2, 1], -R[2, 0])+rad_psi

    Rzyz =np.array([[np.cos(phi)*np.cos(nu)*np.cos(psi)-np.sin(phi)*np.sin(psi), -np.cos(phi)*np.cos(nu)*np.sin(psi)-np.sin(phi)*np.cos(psi), np.cos(phi)*np.sin(nu)],
                   [np.sin(phi)*np.cos(nu)*np.cos(psi)+np.cos(phi)*np.sin(psi), -np.sin(phi)*np.cos(nu)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.sin(phi)*np.sin(nu)],
                   [-np.sin(nu)*np.cos(psi), np.sin(nu)*np.sin(psi), np.cos(nu)]])

    return Rzyz.transpose()

def track_distances(set_a,set_b):
    def set_to_numpy(set):
        return np.array([set['objX'].to_numpy(), set['objY'].to_numpy(), set['height'].to_numpy()]).transpose()
    Xa = set_to_numpy(set_a)
    Xb = set_to_numpy(set_b)
    distances = []
    for i,elema in enumerate(Xa[::30]):
        dist=1000
        for elemb in Xb[::30]:
            dist2 = np.linalg.norm(elema-elemb)
            if dist2<dist:
                dist=dist2
        distances.append(dist)
    #print('mean distance between tracks=',round(np.mean(distances),2),'m')
    return distances


def reconstruct(set,calib):

    # manipulate Rotationmatrix!
    R1 = changeRotationMatrixDegrees(calib.R1, cam1_deg,-cam1_deg,0)
    R2 = changeRotationMatrixDegrees(calib.R2, cam2_deg,-cam2_deg,0)
    tvec1 = np.array([-R1 @ calib.Pos1]).transpose()
    tvec2 = np.array([-R2 @ calib.Pos2]).transpose()
    camintrinsicMatrix1 = np.array(batflight3d.config['camintrinsicMatrix1'], dtype='float32')
    camintrinsicMatrix2 = np.array(batflight3d.config['camintrinsicMatrix2'], dtype='float32')
    calib.Pr1 = camintrinsicMatrix1 @ np.hstack((R1, tvec1))
    calib.Pr2 = camintrinsicMatrix2 @ np.hstack((R2, tvec2))
    calib.Pos1 = (-R1.transpose() @ tvec1).flatten()
    calib.Pos2 = (-R2.transpose() @ tvec2).flatten()

    DSP = set['DSP']
    reconPTS = batflight3d.reconstruct3D(calib, DSP)
    set['reconPTS'] = reconPTS
    errorvector = reconPTS - DSP[['objX', 'objY', 'height']].to_numpy()  # reconPTS - objPTS
    #errorvectors.append(errorvector)
    set['error'] = np.linalg.norm(errorvector, axis=1)
    #print('error: ',set['error'])
    set['camcenter'] = calib.Pos1+0.5*(calib.Pos2-calib.Pos1)
    set['recon_dist_from_camcenter'] = np.linalg.norm(reconPTS - set['camcenter'], axis=1)
    means.append(np.mean(set['recon_dist_from_camcenter']))
    maxes.append(np.max(set['recon_dist_from_camcenter']))

    set['cam1_geoangle'] = batflight3d.vectorToGeoAngle(calib.Dir1)
    set['cam2_geoangle'] = batflight3d.vectorToGeoAngle(calib.Dir2)
    set['cam_distance'] = np.linalg.norm(calib.Pos1 - calib.Pos2)
    set['Pos1'] = calib.Pos1
    set['Pos2'] = calib.Pos2


    dist_from_camcenter.append(set['recon_dist_from_camcenter'])
    error.append(set['error'])
    color.append('#cfffdb')

ddistances =[]
j = 0
def process_DRONE_SIM_PTS(subsets):
    global j
    DSPconc=[]
    DSPs = [d['DSP'] for d in subsets]
    if len(DSPs)==4:
        DSPconc.append(pd.concat([DSPs[1], DSPs[2], DSPs[3]])) # for Testset 0
        DSPconc.append(pd.concat([DSPs[0], DSPs[2], DSPs[3]])) # for Testset 1
        DSPconc.append(pd.concat([DSPs[0], DSPs[1], DSPs[3]])) # for Testset 2
        DSPconc.append(pd.concat([DSPs[0], DSPs[1], DSPs[2]])) # for Testset 3

    elif len(DSPs)==2:
        DSPconc.append(DSPs[1]) # for Testset 0
        DSPconc.append(DSPs[0]) # for Testset 1

    for i, elem in enumerate(DSPconc):
        print('Calculate 3D Calibration, 3D Reconstruction, 3D Errors of Flight ID',j)
        # assign config of clicked flight ID to batflight3d-module
        batflight3d.config = sets[j]['config']
        batflight3d.convert_ref_latlon_to_gps_offset_in_meters()

        #ddistances.append(track_distances(DSPs[i],DSPconc[i]))
        calib_current = calibrate(elem)

        sets[j]['calib'] = calib_current
        sets[j]['DSPconc'] = elem
        j+=1

        reconstruct(subsets[i], calib_current)  # subsets[i]: Pointset to reconstruct


degs = np.linspace(0,2,10)
overall_error = []
cam1_deg = 0
cam2_deg = 0
def process_DSP():
    global mean_errors,error,DSP, bins, bin_median
    error = []
    mean_errors = []
    DSP=[]
    process_DRONE_SIM_PTS(sets[0:4])
    process_DRONE_SIM_PTS(sets[4:8])
    process_DRONE_SIM_PTS(sets[8:10])
    process_DRONE_SIM_PTS(sets[10:12])
    process_DRONE_SIM_PTS(sets[12:14])
    process_DRONE_SIM_PTS(sets[14:16])
    mean_errors = []
    for err in error:
        mean_errors.append(np.mean(err))
    overall_error.append(np.mean(mean_errors))

    # overall distances from camera system:
    all_distances = [i for li in [set['recon_dist_from_camcenter'] for set in sets] for i in li]


    all_errs = [i for li in error for i in li]
    mean_dist_f_c = np.mean(all_distances)
    max_dist_f_c = max(all_distances)
    print('overall mean dist from camera system = ',mean_dist_f_c)
    means.insert(0,mean_dist_f_c)
    maxes.insert(0, max_dist_f_c)

    bins = np.arange(40,261,20)
    bin_indices = np.digitize(np.array(all_distances),bins)
    bin_counts, bin_edges = np.histogram(all_distances, bins=bins)
    bin_median = [np.median(np.array(all_errs)[bin_indices == i]) for i in range(1, len(bins))]


    # add total errors
    error.insert(0,np.array([i for li in error for i in li]))

    # Total Drone Times:
    total_times = [(d['DSP']['time'].iloc[-1]-d['DSP']['time'].iloc[0])/60 for d in sets]

process_DSP()

## Beschriftung
# Trainset
# Testset
name=[]
colors=[]
name.append('All')
colors.append('deeppink')

name.append('2,3,4\n1')
name.append('1,3,4\n2')
name.append('1,2,4\n3')
name.append('1,2,3\n4')
colors.append('pink')
colors.append('pink')
colors.append('pink')
colors.append('pink')

name.append('6,7,8\n5')
name.append('5,7,8\n6')
name.append('5,6,8\n7')
name.append('5,6,7\n8')
colors.append('lightblue')
colors.append('lightblue')
colors.append('lightblue')
colors.append('lightblue')

name.append('10\n9')
name.append('9\n10')
name.append('12\n11')
name.append('11\n12')
name.append('14\n13')
name.append('13\n14')
name.append('16\n15')
name.append('15\n16')

colors.append('lightgreen')
colors.append('lightgreen')
colors.append('paleturquoise')
colors.append('paleturquoise')
colors.append('wheat')
colors.append('wheat')
colors.append('darksalmon')
colors.append('darksalmon')


### PLOTS

## Plot: Errors dependent on the distance from the camera system
fig7, ax7 = plt.subplots()
fig7.canvas.manager.set_window_title('3D Errors per Distance from the Camera System')
#ax8 = ax7.twinx()
ax7.grid()
ax7.bar(bins[:-1], bin_median, width=20, align='edge', edgecolor='black', label='median 3D error')
ax7.set_xlabel('drone distance from camera system / m')
ax7.set_ylabel('median 3D error / m')

# overlay: theoretical curve
f = 20e-3 # focal length in meters
b = 15 # basis in meters
a = 17e-6 # pixel size in meters
x_theoretical = np.linspace(0,270,270)
y_theoretical = x_theoretical**2/((f/a)*b+x_theoretical)
ax7.plot(x_theoretical,y_theoretical,label='voxel depth dimension', color='firebrick')
ax7.legend()

#ax8.plot(bins[:-1]+10,bin_counts,color='red')

## Main Error Plot

# on_click:
def on_click(event):
    val = round(event.xdata-1)
    if val>=1 and val<=16:
        plotFlight(val)

fig = plt.figure('3D Errors per Flight ID')
fig.canvas.mpl_connect('button_press_event', on_click)
ax2 = fig.add_axes([.08,.15,.80,.7])
ax2.set_title('Please click on a Boxplot to open the corresponding 3D Flightpath\n and a Speed Plot. Also note the terminal output information.')
fsize = 13

ax3 = ax2.twinx()
color = 'tab:blue'
ax3.set_ylabel('drone distance\nfrom camera system', color=color, fontsize=fsize)  # we already handled the x-label with ax1
ax3.plot(range(1,len(means)+1),means, color=color,linestyle='--',linewidth=1,marker='.',markersize=10)
ax3.plot(range(1,len(means)+1),maxes, color=color,linestyle='--',linewidth=1,marker='.',markersize=10)
ax3.set_ylim(0,max(maxes))
ax3.tick_params(axis='y', labelcolor=color)

ax2.grid(color='gray', linestyle='-', linewidth=.5, axis='y', which='major', alpha=.3)
box = ax2.boxplot(error, whis=20, notch=True, patch_artist=True)

for patch, color in zip(box['boxes'],colors):
    patch.set_facecolor(color)

names=['All','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
ax2.set_xticklabels(names)
ax2.set_ylabel('3D error in m', fontsize=fsize)
ax2.set_xlabel('flight ID', fontsize=fsize)
ax2.tick_params(axis='both', labelsize=fsize) #labelbottom=False, bottom=True
ax2.set_yticks(np.arange(0,10))


## 3D plots

def plotFlight(setnr):
    global RC

    # print Calibration_Info
    print()
    print('Calibration Info for gray drone flight path:')
    sets[setnr-1]['calib'].printCalibrationInfo()


    RC = batflight3dGUI.plot3D('Flight ID '+str(setnr)+' (for reconstruction and evaluation)')
    t_obj_conc = sets[setnr - 1]['DSPconc']['time'].to_numpy()
    xyz_obj_conc = sets[setnr - 1]['DSPconc'][['objX', 'objY', 'height']].to_numpy()

    xyz = sets[setnr-1]['reconPTS']

    t_obj = sets[setnr - 1]['DSP']['time'].to_numpy()
    xyz_obj = sets[setnr - 1]['DSP'][['objX', 'objY', 'height']].to_numpy()

    RC.plotGPSPoints(xyz_obj_conc,t_obj_conc,sets[setnr-1]['DSPconc'][['lat','lon']].to_numpy(),color='gray',size='1',label='Drone Points '+ sets[setnr-1]['calibname'] +': 3D Calibration')
    t = sets[setnr-1]['DSP']['time'].to_numpy()
    nr = sets[setnr-1]['DSP']['nr'].to_numpy()
    RC.plotPoints(sets[setnr-1]['calib'], xyz ,t,nr,color='cornflowerblue',size='3',label='Image Points '+str(setnr)+': Reconstruction based on 3D Calibration')
    RC.plotGPSPoints(xyz_obj,t_obj,sets[setnr-1]['DSP'][['lat','lon']].to_numpy(),color='black',size='1',label='Drone Points '+str(setnr)+': Evaluation')

    # calculate speed of drone GPS and image reconstruction of drone
    t_drone, speed_drone = batflight3d.calcSpeed(t_obj,xyz_obj,batflight3d.config['speed_plot_z_frames_difference_quotient'])
    t_recon, speed_recon = batflight3d.calcSpeed(t, xyz, batflight3d.config['speed_plot_z_frames_difference_quotient'])

    fig,ax = plt.subplots()
    ax.set_xlabel('t / s')
    ax.set_ylabel('v / km/h')
    ax.plot(t_drone,speed_drone,linewidth=2,label='Drone GPS',color='black')
    ax.plot(t_recon, speed_recon, linewidth=2,label='3D Reconstruction',color='cornflowerblue')
    ax.set_title('Speed of Flight ID '+str(setnr))
    ax.grid()
    ax.legend()
    plt.show()

plt.show()








