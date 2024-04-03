import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import batflight3d

class plot3D:
    def __init__(self,title,*args):
        self.title = title
        self.fig = plt.figure(title,figsize=(8,8))
        self.fig.tight_layout()
        self.ax = plt.axes(projection='3d')
        h3, = self.ax.plot([], [], [], 'go', markersize=5) # , label='clicked PTS'
        h4, = self.ax.plot([], [], [], 'ro', markersize=6) # , label='point in image plot'
        self.dist_plot, = self.ax.plot([], [], [], 'g-')

    def changeView(self,limits):
        x1, x2, y1, y2, z1, z2 = limits
        self.ax.set_xlim3d(x1, x2)
        self.ax.set_ylim3d(y1, y2)
        self.ax.set_zlim3d(z1, z2)
        self.ax_equal()

    def ax_equal(self):
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        self.ax.set_box_aspect((x_range, y_range, z_range))
        plt.draw()
        #plt.show()

    def plotPoints(self,calib3D,xyz,t,nr,color='green',size=3,label='',ls='',marker='o'):

        color_cams = 'black'
        label_cams = ''
        scale_cams = 1

        pc, = self.ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker=marker, ls=ls, lw=size, markersize=size,label=label,
                      mec=color, mfc=color, color=color)  # 3D Reconstruction
                      #mec=colors[len(self.clouds)], mfc=colors[len(self.clouds)], picker=True)  # 3D Reconstruction

        # current reconPTS used in on_pick:
        self.xyz = xyz
        self.t = t
        self.nr = nr

        Pos1 = calib3D.Pos1
        Pos2 = calib3D.Pos2
        Dir1 = calib3D.Dir1 * 10
        Dir2 = calib3D.Dir2 * 10
        #pc1, = self.ax.plot([Pos1[0], Pos2[0]], [Pos1[1], Pos2[1]], [Pos1[2], Pos2[2]], 'o', markersize=5, mec=colors[len(self.clouds)], mfc=colors[len(self.clouds)])
        pc2 = self.ax.quiver([Pos1[0], Pos2[0]], [Pos1[1], Pos2[1]], [Pos1[2], Pos2[2]], [Dir1[0], Dir2[0]], [Dir1[1], Dir2[1]],
                             [Dir1[2], Dir2[2]], color=color_cams, label=label_cams, length=scale_cams)
                  #[Dir1[2], Dir2[2]], color=colors[len(self.clouds)])



        self.legend = self.ax.legend()
        self.ax_equal() # statt canvas_draw

    def plotPointsColored(self,calib3D,xyz,t,nr, points_flightpathnr):
        colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.BASE_COLORS)
        colors = np.array(colors * 1000)

        pc = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', s = 1,c=colors[points_flightpathnr])  # 3D Reconstruction
        #pc = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=t, cmap='spring', picker=True,
        #                     label=str(len(self.legend.texts)) + ' colored')  # 3D Reconstruction

        # current reconPTS used in on_pick:
        self.xyz = xyz
        self.t = t
        self.nr = nr
        self.points_flightpathnr = points_flightpathnr


        Pos1 = calib3D.Pos1
        Pos2 = calib3D.Pos2
        Dir1 = calib3D.Dir1 * 10
        Dir2 = calib3D.Dir2 * 10
        #pc1, = self.ax.plot([Pos1[0], Pos2[0]], [Pos1[1], Pos2[1]], [Pos1[2], Pos2[2]], 'o', markersize=5, mec=colors[len(self.clouds)], mfc=colors[len(self.clouds)])
        pc2 = self.ax.quiver([Pos1[0], Pos2[0]], [Pos1[1], Pos2[1]], [Pos1[2], Pos2[2]], [Dir1[0], Dir2[0]], [Dir1[1], Dir2[1]],
                  [Dir1[2], Dir2[2]], color=colors[0])

        self.legend = self.ax.legend()
        self.ax_equal()


    def plotGPSPoints(self,objPTS,objPTS_time,objPTS_latlon,color='black',size=1,label='Drone'):

        self.xyz_obj = objPTS
        self.t_obj = objPTS_time
        self.latlon_obj = objPTS_latlon

        h1, =  self.ax.plot(self.xyz_obj[:,0], self.xyz_obj[:,1], self.xyz_obj[:,2], 'o', markersize=size, mec=color, mfc=color, label=label)
        h2, = self.ax.plot([],[],[], 'bo', markersize=0.3)

        self.legend = self.ax.legend()
        self.ax_equal()


def synchronizationPlot(meandistances_meters,title,vline=0):
    fig, ax = plt.subplots()
    y = 1/np.array(meandistances_meters)
    ax.plot(y)
    ax.set_title(title)
    ax.set_ylabel('1 / 3D error\n\\m')
    ax.set_xlabel('drone signal offset\n\\steps')
    ax.grid()
    if vline>0:
        ax.vlines(x=vline,ymin=min(y), ymax=max(y), color='red')

class plot:
    def __init__(self,x,y,color,lw,xlabel,ylabel,label='',title=''):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.plot(x,y,linewidth=lw,label=label,color=color)

        if title != '':
            self.ax.set_title(title)
        self.ax.grid()
        if label!='':
            self.ax.legend()
    def show(self):
        self.ax.show()

    def plot(self,x,y,color,lw,label=''):
        self.ax.plot(x, y, linewidth=lw, label=label, color=color)
        if label != '':
            self.ax.legend()

def show():
    plt.show()