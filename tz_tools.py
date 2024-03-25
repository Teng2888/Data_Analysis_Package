#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:48:23 2019

@author: tengzhang
"""
#It is data analysis tools for the diving board measurement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import scipy.constants as con
from qcodes.dataset import load_by_id
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter,LinearLocator
from qcodes import (
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_experiment,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
from scipy.signal import find_peaks
from scipy import optimize
from scipy import constants as con
from scipy.stats import linregress
from scipy.signal import savgol_filter
from uncertainties import ufloat
from uncertainties.umath import *
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import math
import shutil
import os

#The following code is copyied from Tailung Wu. He has the copyright.
def Load_Q(filename,Cols):
    """
    Load data from QCodes program into a pandas data frame\n
    Parameters:\n
    ===========\n
    Filename: Qcodes data file\n
    Cols : array of column names\n
    \n
    Return Pandas dataframe \n
    """
    df=pd.read_csv(filename,sep='\t',skiprows=[0,2],header=0)
    #df.rename(columns={df.columns[0]: df.columns[0][3:-1]}, inplace=True)
    df.columns=Cols
    return df

def findnp_x(df,x): #find the nearest point for the x direction
   df = reshape(df)
   Xn = countingx(df)

   for n in range(1,Xn):
       if n == Xn-1:
          x_out = df.iloc[Xn-1,1]
          break

       dx_m = np.abs(x - df.iloc[n,1])
       dx_b = np.abs(x - df.iloc[n-1,1])
       dx_a = np.abs(x - df.iloc[n+1,1])

       if dx_m < dx_b and dx_m < dx_a:
           x_out = df.iloc[n,1]
           break
       else:
           pass

   return x_out

def findnp_y(df,y): #find the nearest point for the y direction
   df = reshape(df)
   N = int(len(df.iloc[:,0]))
   Xn = countingx(df)

   for n in range(0,N,Xn):
       if n == N-Xn:
          y_out = df.iloc[n,0]
          break

       dy_m = np.abs(y - df.iloc[n,0])
       dy_b = np.abs(y - df.iloc[n-Xn,0])
       dy_a = np.abs(y - df.iloc[n+Xn,0])

       if dy_m < dy_b and dy_m < dy_a:
           y_out = df.iloc[n,0]
           break
       else:
           pass
   return y_out

def linecut(df, v, axis = 'y'): #find the cut line for the a 2D plot in x or y direction
    df = reshape(df)
    if axis == 'y':
       yv = findnp_y(df, v)
       dftemp = df.loc[df.iloc[:,0] == yv]
       dftemp = dftemp.reset_index(drop = True)
       cols = list(dftemp)
       cols[0], cols[1], cols[2] = cols[1], cols[2], cols[0]
       dftemp = dftemp.loc[:,cols]
       fig, ax = plot_1(dftemp)
       return dftemp, fig, ax
    elif axis == 'x':
        xv = findnp_x(df, v)
        dftemp = df.loc[df.iloc[:,1] == xv]
        dftemp = dftemp.reset_index(drop = True)
        cols = list(dftemp)
        cols[0], cols[1], cols[2] = cols[0], cols[2], cols[1]
        dftemp = dftemp.loc[:,cols]
        fig, ax = plot_1(dftemp)
        return dftemp, fig, ax
    else:
        return print('Error: Pls choose x OR y as an axis!')


def countingx(df):
    N = int(len(df.iloc[:,0]))
    Xtemp = df.iloc[0,1]

    Xn=1
    for n in range(1,N):
        if df.iloc[n,1] != Xtemp:
            Xn=Xn+1
        else:
            break
    return Xn

def reshape(df): #check whether the last line is finished or not
    N = int(len(df.iloc[:,0]))
    Xtemp = df.iloc[0,1]

    Xn=1
    for n in range(1,N):
        if df.iloc[n,1] != Xtemp:
            Xn=Xn+1
        else:
            break


    if N%Xn !=0:
        l = int(N%Xn)
        df_out = df.iloc[:-l]
    else:
        df_out = df
    return df_out

def plot(df, zmin,zmax, zcenter, change_order = False, color_map = 'viridis'):
    df = reshape(df)

    #plt.rcParams['figure.figsize'] = [10, 8]
    N = int(len(df.iloc[:,0]))
    Ytemp = df.iloc[0,0]
    Yn=0

    for n in range(0,N):   #count number of values in X-axis.
        if df.iloc[n,0] != Ytemp:
            Yn = Yn+1
            Ytemp=df.iloc[n,0]

    Yn=Yn+1
    Xn = int(N/(Yn)) #calculate the number of in Y-axis
    #found the limit of the data point
    yin = df.iloc[0,0]
    xin = df.iloc[0,1]
    yfin = df.iloc[N-1,0]
    xfin = df.iloc[Xn-1,1]
    #define the grid of the plot
    xarray = np.linspace(xin, xfin, Xn)
    yarray = np.linspace(yin, yfin, Yn)
    xx, yy = np.meshgrid(xarray, yarray)

    data = np.empty(shape = [Yn,Xn])

    for n in range(N):
        x = n//Xn
        y = n%Xn
        data[x,y] = df.iloc[n,2]

    if change_order == False:
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(xx, yy, data, vmin=zmin, vmax=zmax, cmap = color_map,linewidth=0,rasterized=True)

        #norm = mc.Normalize(vmin=zmin, vmax=zmax)
        norm = mc.TwoSlopeNorm(vcenter=zcenter, vmin=zmin, vmax=zmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cax.cmap)
        sm.set_array([])
        cax = fig.colorbar(sm)

        ax.set_ylabel(df.columns[0],fontsize= 20)
        ax.set_xlabel(df.columns[1],fontsize= 20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        cax.set_label(df.columns[2], rotation=270, labelpad=30, fontsize=20)
        cax.ax.tick_params(axis='y', which='major', direction='out', labelsize= 20)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    else:
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(yy, xx, data, vmin=zmin, vmax=zmax, cmap = color_map,linewidth=0,rasterized=True)

       #norm = mc.Normalize(vmin=zmin, vmax=zmax)
        norm = mc.TwoSlopeNorm(vcenter=zcenter, vmin=zmin, vmax=zmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cax.cmap)
        sm.set_array([])
        cax = fig.colorbar(sm)

        ax.set_ylabel(df.columns[1],fontsize= 20)
        ax.set_xlabel(df.columns[0],fontsize= 20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        cax.set_label(df.columns[2], rotation=270, labelpad=30, fontsize=20)
        cax.ax.tick_params(axis='y', which='major', direction='out', labelsize= 20)
        fig.tight_layout()

    return fig, ax

def plot_1(df): #for normal 1D plot
    #plt.rcParams['figure.figsize'] = [12,10 ]
    fig, ax = plt.subplots()
    ax.plot(df.iloc[:,0],df.iloc[:,1])
    ax.set_xlabel(df.columns[0],fontsize= 20)
    ax.set_ylabel(df.columns[1],fontsize= 20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params('both', which='both',direction='in',right = True, top = True )
    plt.tight_layout()
    return fig, ax

def plot_1_multi(df1,label1,df2,label2,df3,label3):
    #plt.rcParams['figure.figsize'] = [12,10 ]
    fig, ax = plt.subplots()
    ax.plot(df1.iloc[:,0],df1.iloc[:,1],label= label1)
    ax.plot(df2.iloc[:,0],df2.iloc[:,1],label= label2)
    ax.plot(df3.iloc[:,0],df3.iloc[:,1],label= label3)
    ax.set_xlabel(df1.columns[0],fontsize= 20)
    ax.set_ylabel(df1.columns[1],fontsize= 20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params('both', which='both',direction='in',right = True, top = True )
    ax.legend()
    plt.tight_layout()
    return fig, ax





def GNS(df):
    df = reshape(df)
    df_1 = df.loc[df['$V_{QPC}(V)$'] == df.iloc[0,0]] #find the center of the tunneling spectra
    G_min = df_1['$G(2e^2/h)$'].min()
    G_min_index = df_1['$G(2e^2/h)$'].idxmin()
    offset = df_1.iloc[G_min_index, 1]

    Nx = countingx(df)
    Ny = int(len(df.iloc[:,0])/Nx)

    G_N = np.empty(shape = Ny)
    G_S = np.empty(shape = Ny)
    idx = Nx

    for n in range(Ny):
        idx = Nx-1 + n*Nx
        G_N_temp = df.iloc[idx, 2]
        G_S_temp = df.iloc[G_min_index +n*Nx, 2]
        G_N[n] = G_N_temp
        G_S[n] = G_S_temp

    return G_N, G_S

def th_GNS():
    G_N_th = np.logspace(-3,1,500)
    G_S_th = (2*G_N_th**2)/(2-G_N_th)**2

    return G_N_th, G_S_th

def th_GNS_thermal(D, t):
    G_N_th = np.logspace(-3,1,500)
    G_S_th = 2*G_N_th**2/(2-G_N_th)**2 + np.sqrt((np.pi * 2 *D)/(86.17 * t* 1E-3)) * np.exp(-D/(86.17 * t* 1E-3))*G_N_th
    return G_N_th, G_S_th

def hall_data(df0T,dfxT,B_perp,W_um, L_um, left_y_lower_limit_kcm2VS,left_y_upper_limit_kcm2VS,right_y_lower_limit_10_12cm_2,right_y_upper_limit_10_12cm_2, tick_num):
    # Input the out-plane field to the calculation. unit:T
    B_perp = B_perp
    # calculated the carry density in unit of m^-2
    dfxT['n_m-2'] = 1/((dfxT['$R_{xy}(\Omega)$']/B_perp)*con.e)
    #assign the carrier density data to the zero field dataframe
    df0T['n_m-2'] = dfxT['n_m-2']
    #%% Input the W and L value for resistivity calculation. Unit in um
    W_um = W_um
    L_um = L_um
    df0T['\rho'] = df0T['$R_{xx}(\Omega)$']/(L_um/W_um)
    # calculate the mobilty in unit of m^2/VS
    df0T['\mu(m^2/VS)'] = 1/(df0T['\rho']*con.e*df0T['n_m-2'])
    # create a new dateframe to store all data we have calculated before.
    dfhall = pd.DataFrame()
    # Putting the data inside
    dfhall['$V_{g}(V)$'] = df0T['$V_{g}(V)$']
    dfhall['$\mu(x10^3cm^2/VS)$'] = df0T['\mu(m^2/VS)']*10
    dfhall['$n(x10^{12} cm^{-2})$'] = df0T['n_m-2']/(10**16)
    dfhall['$\\rho_{xx}(\Omega /square)$'] = df0T['\rho']
    # starting this annoyting plot
    plt.rcParams['figure.figsize'] = [12,10 ]
    fig, ax1 = plt.subplots()
    color1 = 'tab:red'
    ax1.set_xlabel(dfhall.columns[0],fontsize= 15)
    ax1.tick_params(axis='x',direction = 'in')
    ax1.set_ylabel(dfhall.columns[1],fontsize= 15, color=color1)
    muvsg, = ax1.plot(dfhall.iloc[:,0],dfhall.iloc[:,1], color=color1, label = '$\mu$ VS.$V_{g}$')
    ax1.tick_params(axis='y',colors = color1, direction = 'in',labelcolor=color1)
    ax1.set_ylim(left_y_lower_limit_kcm2VS,left_y_upper_limit_kcm2VS)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(dfhall.columns[2],fontsize= 15, color=color)  # we already handled the x-label with ax1
    nvsg, = ax2.plot(dfhall.iloc[:,0],dfhall.iloc[:,2],color=color, label = '$n_{2DEG}$ VS. $V_{g}$')
    ax2.tick_params(axis='y',  direction = 'in',labelcolor=color)
    ax2.set_ylim(right_y_lower_limit_10_12cm_2,right_y_upper_limit_10_12cm_2,)

    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    plt.setp(ax2.spines.values(), visible=False)
    ax1.spines['left'].set_edgecolor('tab:red')

    tick_num = tick_num
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], tick_num))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], tick_num))
    #ax1.yaxis.set_ticks(np.arange(0,250,20))
    #ax2.yaxis.set_ticks(np.arange(0,2.2,20))

    #ax1.locator_params(nbins=20, axis='y')
    #ax2.locator_params(nbins=20, axis='y')

    ax1.grid('both')

    #lines = [muvsg,nvsg]

    #ax2.legend(lines, [l.get_label() for l in lines], loc = 0, fontsize = 20)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return dfhall, fig, ax1, ax2

def top_map(filename, FirstDeviceName, OutputName):
    temp01 = pd.read_excel(filename)
    temp02 = temp01[FirstDeviceName]
    temp02 = pd.Series.to_frame(temp02)
    top_probe = {'1':'1_1', '2':'1_14', '3': '1_2', '4':'1_15', '5':'1_3', '6':'1_16', '7':'1_4',
    '8':'1_17','9':'1_5','10':'1_18','11':'1_6','12':'1_19','13':'1_7','14':'1_20',
    '15':'1_8', '16':'1_21','17':'1_9', '18':'1_22','19':'1_10', '20':'1_23', '21':'1_11', '22':'1_24', '23':'1_12','24':'1_25',
    '27':'2_1','28':'2_14','29':'2_2','30':'2_15', '31':'2_3', '32':'2_16', '33':'2_4','34':'2_17','35':'2_5','36':'2_18','37':'2_6',
    '38':'2_19', '39':'2_7', '40':'2_20','41':'2_8', '42':'2_21','43':'2_9','44':'2_22','45':'2_10','46':'2_23','47':'2_11','48':'2_24','49':'2_12','50':'2_25',
    '52':'3_1','53':'3_14','54':'3_2','55':'3_15','56':'3_3','57':'3_16','58':'3_4','59':'3_17','60':'3_5','61':'3_18','62':'3_6','63':'3_19',
    '64':'3_7','65':'3_20','66':'3_8','67':'3_21','68':'3_9','69':'3_22','70':'3_10','71':'3_23','72':'3_11',
    '73':'3_24','74':'3_12','75':'3_25','76':'4_1','77':'4_14','78':'4_2','79':'4_15','80':'4_3','81':'4_16','82':'4_4','83':'4_17','84':'4_5',
    '85':'4_18','86':'4_6','87':'4_19','88':'4_7','89':'4_20','90':'4_8','91':'4_21','92':'4_9','93':'4_22','94':'4_10','95':'4_23','96':'4_11','97':'4_24','98':'4_12','99':'4_25'}
    temp02[FirstDeviceName] = temp02[FirstDeviceName].astype(str)
    temp02['top_probe'] = temp02[FirstDeviceName].map(top_probe)
    temp02.to_excel(OutputName)


def bottom_map(filename, FirstDeviceName, OutputName):
    temp01 = pd.read_excel(filename)
    temp02 = temp01[FirstDeviceName]
    temp02 = pd.Series.to_frame(temp02)

    bottom_probe = {'1':'1_25', '2':'1_12', '3': '1_24', '4':'1_11', '5':'1_23', '6':'1_10', '7':'1_22',
    '8':'1_9','9':'1_21','10':'1_8','11':'1_20','12':'1_7','13':'1_19','14':'1_6',
    '15':'1_18', '16':'1_5','17':'1_7', '18':'1_4','19':'1_16', '20':'1_3', '21':'1_15', '22':'1_2', '23':'1_14','24':'1_1',
    '27':'2_25','28':'2_12','29':'2_24','30':'2_11', '31':'2_23', '32':'2_10', '33':'2_22','34':'2_9','35':'2_21','36':'2_8','37':'2_20',
    '38':'2_7', '39':'2_19', '40':'2_6','41':'2_18', '42':'2_5','43':'2_17','44':'2_4','45':'2_16','46':'2_3','47':'2_15','48':'2_2','49':'2_14','50':'2_1',
    '52':'3_25','53':'3_12','54':'3_24','55':'3_11','56':'3_23','57':'3_10','58':'3_9','59':'3_22','60':'3_21','61':'3_8','62':'3_20','63':'3_7',
    '64':'3_19','65':'3_6','66':'3_18','67':'3_5','68':'3_17','69':'3_4','70':'3_16','71':'3_3','72':'3_15',
    '73':'3_2','74':'3_14','75':'3_1','76':'4_25','77':'4_12','78':'4_24','79':'4_11','80':'4_23','81':'4_10','82':'4_22','83':'4_9','84':'4_21',
    '85':'4_8','86':'4_20','87':'4_7','88':'4_19','89':'4_6','90':'4_18','91':'4_5','92':'4_17','93':'4_4','94':'4_16','95':'4_3','96':'4_15','97':'4_2','98':'4_14','99':'4_1'}

    temp02[FirstDeviceName] = temp02[FirstDeviceName].astype(str)
    temp02['bottom_probe'] = temp02[FirstDeviceName].map(bottom_probe)
    temp02.to_excel(OutputName)

#The following 2D plot function is Tailung Wu's code. He has the copyright.
def XYZColorMap(DataSet
                ,z=2,x=0,y=1
                ,Zmin=None,Zmax=None,Zstep=100
                ,title=''
                ,zlabel='',xlabel='',ylabel=''
                ,N=200 #contour N automatically-chosen levels.
                ,color=plt.cm.viridis):
    """
    X,Y,Z Color coutour plot\n
    Parameters:\n
    ===========\n
    DataSet: Pandas dataframe\n
    Zmin (None), Zmax (None), Zstep=100
    z : Column index for z (default : 2)\n
    x : Column index for x (default : 0)\n
    y : Column index for y (default : 1)\n
    title:\n
    xlabel,ylabel,zlabel:\n
    N:contour N automatically-chosen levels.
    color: \n
    """
    if Zmax == None:
        Zmax = DataSet.iloc[:,z].values.max()

    if Zmin == None:
        Zmin = DataSet.iloc[:,z].values.min()

    levels = np.linspace(Zmin,Zmax, num=100)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    cax=ax.tricontourf(DataSet.iloc[:,x],DataSet.iloc[:,y],DataSet.iloc[:,z]
                    ,N,cmap=color
                    ,vmax=Zmax, vmin=Zmin
                    ,extend = 'both'
                    ,levels=levels)

    # Axis label
    if xlabel == '':
        xlabel = DataSet.columns[x]
    if ylabel == '':
        ylabel = DataSet.columns[y]
    if zlabel == '':
        zlabel = DataSet.columns[z]
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)

    #Color bar
    cbar=fig.colorbar(cax)
    cbar.set_label(zlabel,rotation=270 ,labelpad=14 ,fontsize = 14)
    cbar.ax.tick_params(axis='y', direction='in')

    #Axis tick
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    #plt title
    if title != '' :
        ax.set_title(title, style='italic')

    #Data Cursor
    #dc1 = datacursor(ax, xytext=(15, -15), bbox=None)

     # set useblit = True on gtkagg for enhanced performance
    #Cursor(ax,useblit=True, color='black', linewidth=0.5,linestyle='--')

    #fig.savefig('plot.png', dpi=600,transparent=True)

    return fig, ax
    
def load_db_as_df(runID, para_name, cols):
    #runID = int(runID)
    dataset = load_by_id(runID)
    dataframe = dataset.get_data_as_pandas_dataframe()[para_name]
    dataframe = dataframe.reset_index()
    dataframe.columns = cols
    return dataframe

def hall_load_db_as_df(runID_0T,runID_xT, para_Rxx, para_Rxy):
    #runID = int(runID)
    cols_Rxx = ['$V_{g}(V)$', '$R_{xx}(\\Omega)$']
    cols_Rxy = ['$V_{g}(V)$', '$R_{xy}(\\Omega)$']
    df0T = load_db_as_df(runID_0T,para_Rxx,cols_Rxx)
    dfxT = load_db_as_df(runID_xT,para_Rxy,cols_Rxy)
    return df0T, dfxT


###############################################################
#######below is the code for the SdHO Analysis
def sin(x, a, T):
    return a * np.sin(2*np.pi*x/T)

#This is the package for changing B to 1/B and set up the range
def invB(df_linecut, i, f):
    df_temp1 = df_linecut[(df_linecut.iloc[:, 0]>=i)&(df_linecut.iloc[:, 0]<=f)]
    df_temp = pd.DataFrame()
    df_temp['$1/B$(1/T)'] = 1/df_temp1.iloc[:, 0]
    df_temp['$R_{xx}$(ohm)'] = df_temp1.iloc[:, 1]
    df_temp = df_temp.reset_index(drop = True)
    return df_temp

# This package for removing background
def nobackgroundandNormal(df):
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    
    peaks, _=  find_peaks(y)
    y_dip = -y
    dips, _=  find_peaks(y_dip)
    
    #Find the fit line for peaks
    a1, b1, c1, d1, e1 = np.polyfit(x[peaks], y[peaks],4)
    #For dips
    a2, b2, c2, d2, e2 = np.polyfit(x[dips], y[dips],4)
 
    #define the function
    fy_peak = a1*x**4 + b1*x**3 + c1*x**2 + d1*x + e1
    fy_dip = a2*x**4 + b2*x**3 + c2*x**2 + d2*x + e2
    
    #remove the back ground
    fbg = (fy_peak + fy_dip)/2
    dy = y - fbg
    
    #Find the amp for the normalized
    Amp = (fy_peak - fy_dip)/2
    dy_normalized = dy/ Amp
    
    return x, dy_normalized
    
#here is the fitting for the sine
def sinfit(x, y):
    params, params_covariance = optimize.curve_fit(sin, x, y,
                                               p0=[1, 0.05])
    #return amplitude, period, error for the amplitude and period
    perr = np.diag(params_covariance)
    return params[0], params[1], np.sqrt(perr[0]), np.sqrt(perr[1])

#calculate the SdH density
def n_SdH(T, Terr):
    n = 2 * con.e / (con.h * T)
    nerr = (2 * con.e / ((con.h * T)**2)) *(Terr*con.h)
    return n, nerr
    
#combine everything together
def SdH0(df_linecut, i, f):
    df_temp = invB(df_linecut, i, f)
    x, y = nobackgroundandNormal(df_temp)
    A, T, Aerr, Terr = sinfit(x, y)
    n, nerr = n_SdH(T, Terr)
    n_cm = n/1e16
    nerr_cm = nerr/1e16
    return n_cm, nerr_cm

def linecut_wo_plt(df, v, axis = 'y'): #find the cut line for the a 2D plot in x or y direction
    df = reshape(df)
    if axis == 'y':
       yv = findnp_y(df, v)
       dftemp = df.loc[df.iloc[:,0] == yv]
       dftemp = dftemp.reset_index(drop = True)
       cols = list(dftemp)
       cols[0], cols[1], cols[2] = cols[1], cols[2], cols[0]
       dftemp = dftemp.loc[:,cols]
       return dftemp
    elif axis == 'x':
        xv = findnp_x(df, v)
        dftemp = df.loc[df.iloc[:,1] == xv]
        dftemp = dftemp.reset_index(drop = True)
        cols = list(dftemp)
        cols[0], cols[1], cols[2] = cols[0], cols[2], cols[1]
        dftemp = dftemp.loc[:,cols]
        return dftemp
    else:
        return print('Error: Pls choose x OR y as an axis!')
        
###############################################################
#######below is the code for the SdHO Analysis with a simple linear fit
######################################################
def linearfit_T(df):
    x = df.iloc[:, 0].to_numpy()
    yraw= df.iloc[:, 1].to_numpy()
    
    y_smooth = savgol_filter(yraw, 15, 3)
    peaks, _=  find_peaks(y_smooth)
    xpeak = x[peaks]
    N = int(len(xpeak))
    xN = np.empty(N)
    for x in range(1, N):
        xN[x] = x

    paras = linregress(xN,xpeak)

    return paras[0], paras[2]

def linearfit_T_dips(df):
    x = df.iloc[:, 0].to_numpy()
    yraw= df.iloc[:, 1].to_numpy()

    y_smooth_neg = savgol_filter(yraw, 15, 3)
    y_smooth = -y_smooth_neg
    peaks, _=  find_peaks(y_smooth)
    xpeak = x[peaks]
    N = int(len(xpeak))
    xN = np.empty(N)
    for x in range(1, N):
        xN[x] = x

    paras = linregress(xN,xpeak)

    return paras[0], paras[2]

#calculate the SdH density
def n_SdH_woerr(T):
    n = 2 * con.e / (con.h * T)
    return n
    
#linear fit
#combine everything together
def SdH0_linear(df_linecut, i, f):
    df_temp = invB(df_linecut, i, f)
    m,r = linearfit_T(df_temp)
    n= n_SdH_woerr(m)
    n_cm = n/1e16
    return n_cm

# this is the code for the auto analysis for SdHo for the landau fan linear
def SdH_density(df, Vg_i, Vg_f, Bi, Bf):
    df = reshape(df)
    B_temp = df.iloc[0,0]
    df_temp1 = linecut_wo_plt(df, B_temp, 'y')
    
    df_temp2 = df_temp1[(df_temp1.iloc[:, 0]<=Vg_i)&(df_temp1.iloc[:, 0]>=Vg_f)]
    Vg_array = df_temp2.iloc[:, 0].to_numpy()
    
    N = int(len(df_temp2.iloc[:,0]))
    n = np.empty(N)
    vg = np.empty(N)
    
    for x in range(N):
        vg[x] = Vg_array[x]
        df_temp3= linecut_wo_plt(df, vg[x], 'x')
        n[x] = SdH0_linear(df_temp3, Bi, Bf)
    
    df_final = pd.DataFrame()
    df_final['$V_{g}(V)$'] = vg
    df_final['$n_{SdH}(x10^{12} cm^{-2})$'] = n
    return df_final

def slope_density(df_Rxy, Bperp):
    df_line_cut = linecut_wo_plt(df_Rxy,Bperp,'y')
    B_perp = np.abs(df_line_cut.iloc[0,2])
    df_density = pd.DataFrame()
    df_density['$V_{g}$(V)'] = df_line_cut.iloc[:,0]
    df_density['$n(x10^{12} cm^{-2})$'] = (1/((df_line_cut.iloc[:,1]/B_perp)*con.e))/(10**16)
    return df_density

def extract_background(x, yhat):
    peaks, _=  find_peaks(yhat)
    y_dip = -yhat
    dips, _=  find_peaks(y_dip)
    #Find the fit line for peaks
    a1, b1, c1, d1, e1 = np.polyfit(x[peaks], yhat[peaks],4)
    #For dips
    a2, b2, c2, d2, e2 = np.polyfit(x[dips], yhat[dips],4)
    #define the function
    fy_peak = a1*x**4 + b1*x**3 + c1*x**2 + d1*x + e1
    fy_dip = a2*x**4 + b2*x**3 + c2*x**2 + d2*x + e2
    fbg = (fy_peak + fy_dip)/2
    dy =  yhat - fbg
    return dy
    
def linearfit_T_wo_smooth(df):
    x = df.iloc[:, 0].to_numpy()
    yraw= df.iloc[:, 1].to_numpy()
    y = -yraw
    peaks, _=  find_peaks(y)
    xpeak = x[peaks]
    N = int(len(xpeak))
    xN = np.empty(N)
    for x in range(1, N):
        xN[x] = x

    paras = linregress(xN,xpeak)

    return paras[0], paras[2]

def SdH_density_wo_smooth(df, Vg_i, Vg_f, Bi, Bf):
    df = reshape(df)
    B_temp = df.iloc[0,0]
    df_temp1 = linecut_wo_plt(df, B_temp, 'y')

    df_temp2 = df_temp1[(df_temp1.iloc[:, 0]<=Vg_i)&(df_temp1.iloc[:, 0]>=Vg_f)]
    Vg_array = df_temp2.iloc[:, 0].to_numpy()

    N = int(len(df_temp2.iloc[:,0]))
    n = np.empty(N)
    vg = np.empty(N)

    for x in range(N):
        vg[x] = Vg_array[x]
        df_temp3= linecut_wo_plt(df, vg[x], 'x')
        n[x] = SdH0_linear_wo_smooth(df_temp3, Bi, Bf)

    df_final = pd.DataFrame()
    df_final['$V_{g}(V)$'] = vg
    df_final['$n_{SdH}(x10^{12} cm^{-2})$'] = n
    return df_final

def SdH0_linear_wo_smooth(df_linecut, i, f):
    df_temp = invB(df_linecut, i, f)
    m,r = linearfit_T_wo_smooth(df_temp)
    n= n_SdH_woerr(m)
    n_cm = n/1e16
    return n_cm

def SdH_density_dips(df, Vg_i, Vg_f, Bi, Bf):
    df = reshape(df)
    B_temp = df.iloc[0,0]
    df_temp1 = linecut_wo_plt(df, B_temp, 'y')

    df_temp2 = df_temp1[(df_temp1.iloc[:, 0]<=Vg_i)&(df_temp1.iloc[:, 0]>=Vg_f)]
    Vg_array = df_temp2.iloc[:, 0].to_numpy()

    N = int(len(df_temp2.iloc[:,0]))
    n = np.empty(N)
    vg = np.empty(N)

    for x in range(N):
        vg[x] = Vg_array[x]
        df_temp3= linecut_wo_plt(df, vg[x], 'x')
        n[x] = SdH0_linear_dips(df_temp3, Bi, Bf)

    df_final = pd.DataFrame()
    df_final['$V_{g}(V)$'] = vg
    df_final['$n_{SdH}(x10^{12} cm^{-2})$'] = n
    return df_final

def SdH0_linear_dips(df_linecut, i, f):
    df_temp = invB(df_linecut, i, f)
    m,r = linearfit_T_dips(df_temp)
    n= n_SdH_woerr(m)
    n_cm = n/1e16
    return n_cm
######################################################################
###############Codes below are for the 2-Band Drude model##############
#######################################################################
#find the carrier density at this given gate voltage
def find_y_given_x(df, x):
    Xn = int(len(df))
    
    for n in range(1,Xn):
       if n == Xn-1:
          x_out = df.iloc[Xn-1,0]
          break

       dx_m = np.abs(x - df.iloc[n,0])
       dx_b = np.abs(x - df.iloc[n-1,0])
       dx_a = np.abs(x - df.iloc[n+1,0])

       if dx_m < dx_b and dx_m < dx_a:
           x_out = df.iloc[n,0]
           break
       else:
           pass
        
    y = df[df.iloc[:,0] == x_out].iloc[0,1]
    return y

#This is code to clculate the mobility directly from the 2D landua fan diagram
def mobility_vs_density(df_Rxx, df_Rxy, Bf, W, L):
    df0T = linecut_wo_plt(df_Rxx, 0.001)
    dfxT = linecut_wo_plt(df_Rxy, Bf)
    # Input the out-plane field to the calculation. unit:T
    B_perp = Bf
    # calculated the carry density in unit of m^-2
    dfxT['n_m-2'] = 1/((dfxT.iloc[:,1]/B_perp)*con.e)
    #assign the carrier density data to the zero field dataframe
    df0T['n_m-2'] = dfxT['n_m-2']
    #%% Input the W and L value for resistivity calculation. Unit in um
    W_um = W
    L_um = L
    df0T['\rho'] = df0T.iloc[:,1]/(L_um/W_um)
    # calculate the mobilty in unit of m^2/VS
    df0T['\mu(m^2/VS)'] = 1/(df0T['\rho']*con.e*df0T['n_m-2'])
    # create a new dateframe to store all data we have calculated before.
    dfhall = pd.DataFrame()
    # Putting the data inside
    dfhall['$V_{g}(V)$'] = df0T.iloc[:,0]
    dfhall['$\mu(x10^3cm^2/VS)$'] = df0T['\mu(m^2/VS)']*10
    dfhall['$n(x10^{12} cm^{-2})$'] = df0T['n_m-2']/(10**16)
    dfhall['$\\rho_{xx}(\Omega /square)$'] = df0T['\rho']
    
    return dfhall

#This is the code to do the two band fitting for one given gate volatge
from scipy.optimize import curve_fit
def two_band_linecut(df_xy, df_mobility_curve, df_slope_denisty, B_max_fit, Vg):
    
    global mu2deg
    mu2deg = find_y_given_x(df_mobility_curve, Vg) /10 #find the 2deg mobility at the given gate voltagte
    n2deg_i = find_y_given_x(df_slope_denisty, Vg) *1e16 #find the inital guess value for 2DEG density.
    npar_i = n2deg_i/3 #initial guess value for carrier denisty in the parallel channel
    mup_i = mu2deg/10  #intiall guess value for mobility of the carrier in the parallel channel
    
    df_xy_line_raw = linecut_wo_plt(df_xy, Vg, 'x')
    #select the data from the low B field range to avoid the oscillation 
    df_xy_for_fit= df_xy_line_raw[df_xy_line_raw.iloc[:,0]<B_max_fit]
    
    param_bounds=([0,0,0],[3e16,3e16,1])
    
    def Hall_in (bf,n2deg,npar,mup):
        #n2deg=0.72e15
        e=1.602e-19
        return (bf*(1+mu2deg*mu2deg*bf*bf)*(1+mup*mup*bf*bf)*(n2deg*e*mu2deg*mu2deg*(1+mup*mup*bf*bf)+npar*e*mup*mup*(1+mu2deg*mu2deg*bf*bf)))/(((n2deg*e*mu2deg*(1+mup*mup*bf*bf))+(npar*e*mup*(1+mu2deg*mu2deg*bf*bf)))**2+
                                                                                                                       (((n2deg*e*mu2deg*mu2deg*bf*(1+mup*mup*bf*bf))+(npar*e*mup*mup*bf*(1+mu2deg*mu2deg*bf*bf)))**2))
#ref. Peters et all. "improvements of the transport properties of  a high mobility electron system by intentional parallel channel

    pars, cov = curve_fit(f=Hall_in, xdata = df_xy_for_fit.iloc[:, 0] ,ydata=df_xy_for_fit.iloc[:, 1], 
                    p0=[n2deg_i,npar_i,mup_i], bounds = param_bounds)
    
    return pars, cov

#now combining everything together. two-band analysis for all gate voltages. 
def twoD_two_band_analysis(df_xy, df_mobility_curve, df_slope_denisty, B_max_fit, Vg_i, Vg_f):
    df = reshape(df_xy)

    df_temp2 = df_slope_denisty[(df_slope_denisty.iloc[:, 0]<=Vg_i)&(df_slope_denisty.iloc[:, 0]>=Vg_f)]
    Vg_array = df_temp2.iloc[:, 0].to_numpy()

    N = int(len(df_temp2.iloc[:,0]))
    n_par = np.empty(N)
    n_par_error = np.empty(N)
    mu_par = np.empty(N)
    mu_par_error = np.empty(N)
    n_2deg = np.empty(N)
    n_2deg_error = np.empty(N)
    vg = np.empty(N)

    for x in range(N):
        vg[x] = Vg_array[x]
        pars, cov = two_band_linecut(df_xy, df_mobility_curve, df_slope_denisty, B_max_fit, vg[x])
        n_2deg[x] = pars[0]/1e16
        n_2deg_error[x] = (cov[0,0]**0.5)/1e16
        n_par[x] = pars[1]/1e16
        n_par_error[x] = (cov[1,1]**0.5)/1e16
        mu_par[x] = pars[2]*10
        mu_par_error[x] = (cov[2,2]**0.5)*10

    df_final = pd.DataFrame()
    df_final['$V_{g}(V)$'] = vg
    df_final['$n_{2DEG}(x10^{12} cm^{-2})$'] = n_2deg
    df_final['$n_{parallel}(x10^{12} cm^{-2})$'] = n_par
    df_final['$\mu(x10^3cm^2/VS)$'] = mu_par
    
    df_final['$n_{2DEG} error(x10^{12} cm^{-2})$'] = n_2deg_error
    df_final['$n_{parallel} error(x10^{12} cm^{-2})$'] = n_par_error
    df_final['$\mu error(x10^3cm^2/VS)$'] = mu_par_error
    
    return df_final

#======================================================================
#Codes below are for the analysis of Floquet Josephson Junction data. 
def IV_curve_from_2D_scan(df, fit_range_i, fit_range_f):
    """
    df is the 2D scan: Col_1 is the DC current,\n
    Col_2 is the DC voltage bias for the tunneling spectra,\n
    Col_3 is the DC voltage between two SC leads\n
    The DC voltage measurement offset is extracted from the mean voltage when JJ is superconducting\n
    fit_range_i: inital value of DC current of supercurrent region.
    fit_range_f: final value of the DC current of the supercurrent region. 
    """
    Nx = countingx(df)
    Ny = int(len(df.iloc[:,0])/Nx)
    I_array = np.empty(Ny)
    V_array = np.empty(Ny)

    for index in range(Ny):
        I_array[index] = df.iloc[index*Nx+1,0]
        df_linecut_temp = linecut_wo_plt(df, I_array[index])
        V_array[index] = df_linecut_temp.iloc[:,1].mean()

    df_out = pd.DataFrame()
    df_out['I'] = I_array
    df_out['V'] = V_array #remove the instrument's offset 
    df_average_offset = df_out[(df_out.iloc[:,0]<np.abs(fit_range_f))&(df_out.iloc[:,0]>np.abs(fit_range_i))]
    average_offset = df_average_offset.iloc[:,1].mean()
    df_out['V'] = df_out['V'] - average_offset
    return df_out# %%


#===============================finding peaks function===========================
#===============================small functions for the main function============
def smooth_feature(df, window_i, window_f, window_width, smooth_order):
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    yhat = savgol_filter(y, window_width, smooth_order)
    return x, yhat

def gauss_2(x, A, x0, sigma):
    return H + A / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x-x0)**2 / (2*sigma**2))

def gauss_plot(x, offset, A, x0, sigma):
    return offset + A / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x-x0)**2 / (2*sigma**2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss_2, x, y, p0=[max(y), mean, sigma])
    return popt, pcov

def fitting_peak_for_width (df, center_value, fitting_width):
    global H
    value_i = center_value - fitting_width
    value_f = center_value + fitting_width
    df_short = df[(df.iloc[:,0]>value_i)&(df.iloc[:,0]<value_f)]
    xdata = df_short.iloc[:,0]
    ydata = df_short.iloc[:,1]
    H = 0.95*ydata.min()
    popt, pcov = gauss_fit(xdata, ydata)
    A, x0, sigma = popt
    dA, dx0, dsigma = np.sqrt(np.diag(pcov))
    return popt

#================================the main function============================
def peak_finding_for_Floquet_date(df, 
                                  i_value, 
                                  f_value,
                                  window_i,
                                  window_f,
                                  saperate_value,
                                  guass_fit_range,
                                  smooth_window,
                                  floder_path):
    # combine above code together
    # including plotting function
    """
    df is the 2D scan: Col_1 is the DC current,\n
    Col_2 is the DC voltage bias for the tunneling spectra,\n
    Col_3 is the tunneling conductance of the tunneling probe\n
    i_value (f_value) is the initial(final) value of DC current (y-axis) of the peak extraction process\n
    i_window and f_window are the lower bound and the upper of the DC voltage range where the usuful feature is located\n
    saperate_value is used to set the middle between 4 conductance peaks(2 for each side). It is an important value for locating conductance peaks\n
    guass_fit_range is the DC voltage bias range for fitting each individual conductance peaks\n
    This function generates series of plots for each DC current linecut with the gussian fit curves.\n
    floder_path should be the path of the folder where all those linecuts will be stored.\n
    smooth_window is the data point window for smoothing function to remove all small peaks.\n
    return five numpy arrays: I_arry: DC current value. V_negative_array_2, V_negative_array, V_positive_array, V_positive_array_2 are peaks location (DC voltage bias values)\n
    """
    df_raw = reshape(df)
    df_2D = df_raw[(df_raw.iloc[:,0]>i_value)&(df_raw.iloc[:,0]<f_value)]
    Nx = int(countingx(df_2D))
    Ny = int(len(df_2D.iloc[:,0])/Nx)
    I_array = np.empty(Ny)
    V_negative_array = np.empty(Ny)
    V_positive_array = np.empty(Ny)
    V_negative_array_2 = np.empty(Ny)
    V_positive_array_2 = np.empty(Ny)
    # V_negative_array_error = np.empty(Ny)
    # V_positive_array_error = np.empty(Ny)
    # V_negative_array_2_error = np.empty(Ny)
    # V_positive_array_2_error = np.empty(Ny)
    distance_limit = 20

    for index in range(Ny):
        I_array[index] = df_2D.iloc[index*Nx+1,0]
        print(I_array[index])
        df_linecut_temp = linecut_wo_plt(df_2D, I_array[index])
        n = len(df_linecut_temp.iloc[:,0])-1
        x, yhat = smooth_feature(df_linecut_temp, window_i, window_f, smooth_window, 3)
        x, y_bg = smooth_feature(df_linecut_temp, window_i, window_f, n, 2)
        df_filtered = pd.DataFrame()
        df_filtered['x'] = x
        df_filtered['y'] = yhat/y_bg
        #df_filtered['y'] = yhat
        df_filtered_windowed = df_filtered[(df_filtered['x']>window_i)&(df_filtered['x']<window_f)]
        df_for_peak_finder_1 = df_filtered_windowed[df_filtered_windowed.iloc[:,0]<saperate_value-0.02].reset_index(drop = True)
        df_for_peak_finder_2 = df_filtered_windowed[df_filtered_windowed.iloc[:,0]>saperate_value+0.02].reset_index(drop = True)
        
        x1 = df_for_peak_finder_1.iloc[:,1]
        peaks1, _ = find_peaks(x1, distance = distance_limit)
        #index_target_peak1 = peaks1[-1]
        #index_target_peak3 = peaks1[-2]
        V_initial = np.empty(4)
        for index_2 in range(2):
            V_initial[index_2*2] = df_for_peak_finder_1.iloc[peaks1[int(-1*index_2-1)],0]

        x2 = df_for_peak_finder_2.iloc[:,1]
        peaks2, _ = find_peaks(x2, distance = distance_limit)
        #index_target_peak2 = peaks2[0]
        #index_target_peak4 = peaks2[1]
        for index_3 in range(2):
            V_initial[index_3*2+1] = df_for_peak_finder_2.iloc[peaks2[int(index_3)],0]
        
        peaks_temp = np.empty(4)
        peaks_error_temp = np.empty(4)

        plt.plot(df_linecut_temp.iloc[:,0], df_linecut_temp.iloc[:,1], 'ko', label='data')
        for index_4 in range(4):
            try:
                center_value = V_initial[index_4]
                popt_temp = fitting_peak_for_width(df_linecut_temp, center_value, guass_fit_range)
                peaks_temp[index_4] = popt_temp[1]
                peaks_error_temp[index_4] = popt_temp[2]
                xplot = np.arange(popt_temp[1]-guass_fit_range,popt_temp[1]+guass_fit_range, 0.002)
                plt.plot(xplot, gauss_plot(xplot, H, popt_temp[0],popt_temp[1],popt_temp[2]), '--r',label='Guass fit')
            except:
                pass
        plot_name = str(I_array[index]).replace('.','_')
        plt.ylabel('$G_{top}(2e^2/h)$',fontsize=25)
        plt.xlabel('$V_{T}(m V)$',fontsize=25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(floder_path + plot_name, dpi = 100)
        plt.close()

        V_negative_array[index] = peaks_temp[0]
        V_positive_array[index] = peaks_temp[1]
        V_negative_array_2[index] = peaks_temp[2]
        V_positive_array_2[index] = peaks_temp[3]

    G_upper = df_2D.iloc[:,2].max()
    G_lower = df_2D.iloc[:,2].min()  
    fig,ax = plot(df_raw, G_lower,G_upper, (G_lower+G_upper)/2)
    plt.scatter(V_negative_array, I_array, color = 'red')
    plt.scatter(V_negative_array_2, I_array, color = 'orange')
    plt.scatter(V_positive_array, I_array, color = 'blue')
    plt.scatter(V_positive_array_2, I_array, color = 'green')
    ax.set_ylim(i_value,f_value)
    ax.set_xlim(window_i,window_f)
    plt.ylabel('$I_{DC}(uA)$',fontsize=25)
    plt.xlabel('$V_{T}(m V)$',fontsize=25)
    plt.savefig(floder_path + '2D_scan', dpi = 100)
    plt.close()
    
    return I_array, V_negative_array_2, V_negative_array, V_positive_array, V_positive_array_2


def create_folder(folder_path):
    '''
    创建同名文件夹，如果同名文件夹已存在，则询问用户是否删除旧文件夹，再创建新文件夹。
    '''
    # 如果文件夹已经存在，则询问用户是否删除它
    if os.path.exists(folder_path):
        print(f'文件夹已存在：{folder_path}')
        choice = input('是否删除旧文件夹？(y/n) ')
        if choice.lower() == 'y':
            shutil.rmtree(folder_path)
            print(f'旧文件夹已删除：{folder_path}')
        else:
            print('已取消操作。')
            return

    # 创建新的文件夹
    os.makedirs(folder_path)

    # 输出提示信息
    print(f'文件夹已创建：{folder_path}')

def x_y_array(df_in):
    df = reshape(df_in)
    N = len(df.iloc[:,0])
    Xn = countingx(df)
    Yn = int(N/Xn)
    X_array = np.empty(Xn)
    Y_array = np.empty(Yn)

    for x in range (Xn):
        X_array[x] = df.iloc[x,1]

    for y in range (Yn):
        index_temp = y*Xn+1
        Y_array[y] = df.iloc[index_temp,0]
    
    return X_array, Y_array

def convert_Idc_to_Vdc(df_in, IV_curve_in, min_V, max_V):
    df = reshape(df_in)
    IV_curve = IV_curve_in

    Nx = int(countingx(df))
    Ny = int(len(df.iloc[:,0])/Nx)

    df_out = df.copy(deep = True)

    names = df_out.columns
    
    newname = '$V_{JJ}(uV)$'

    df_out.columns = [newname, names[1], names[2]]


    for y_index in range (Ny):
        x_i = int(y_index * Nx)
        x_f = int((y_index+1) * Nx)
        I_dc_temp = df_out.iloc[x_i+1,0] 
        Vj_temp = find_y_given_x(IV_curve, I_dc_temp)
        df_out.iloc[x_i:x_f, 0] = Vj_temp
    
    df_out = df_out[(df_out.iloc[:,0]>min_V)&(df_out.iloc[:,0]<max_V)]

    return df_out
    

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def plot_2D(df, zmin,zmax, zcenter, change_order = False, color_map = 'viridis'):
    '''
    It is the script for making 2D color map without even distributed data points.
    '''
    df = reshape(df)

    #plt.rcParams['figure.figsize'] = [10, 8]
    N = int(len(df.iloc[:,0]))
    Ytemp = df.iloc[0,0]
    Yn=0

    for n in range(0,N):   #count number of values in X-axis.
        if df.iloc[n,0] != Ytemp:
            Yn = Yn+1
            Ytemp=df.iloc[n,0]

    Yn=Yn+1
    Xn = int(N/(Yn)) #calculate the number of in Y-axis
    #found the limit of the data point
    
    X_array_1, Y_array_1 = x_y_array(df)

    xx, yy = np.meshgrid(X_array_1, Y_array_1)

    data = np.empty(shape = [Yn,Xn])

    for n in range(N):
        x = n//Xn
        y = n%Xn
        data[x,y] = df.iloc[n,2]

    if change_order == False:
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(xx, yy, data, vmin=zmin, vmax=zmax, cmap = color_map,linewidth=0,rasterized=True)

        #norm = mc.Normalize(vmin=zmin, vmax=zmax)
        norm = mc.TwoSlopeNorm(vcenter=zcenter, vmin=zmin, vmax=zmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cax.cmap)
        sm.set_array([])
        cax = fig.colorbar(sm)

        ax.set_ylabel(df.columns[0],fontsize= 20)
        ax.set_xlabel(df.columns[1],fontsize= 20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        cax.set_label(df.columns[2], rotation=270, labelpad=30, fontsize=20)
        cax.ax.tick_params(axis='y', which='major', direction='out', labelsize= 20)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    else:
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(yy, xx, data, vmin=zmin, vmax=zmax, cmap = color_map,linewidth=0,rasterized=True)

       #norm = mc.Normalize(vmin=zmin, vmax=zmax)
        norm = mc.TwoSlopeNorm(vcenter=zcenter, vmin=zmin, vmax=zmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cax.cmap)
        sm.set_array([])
        cax = fig.colorbar(sm)

        ax.set_ylabel(df.columns[1],fontsize= 20)
        ax.set_xlabel(df.columns[0],fontsize= 20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        cax.set_label(df.columns[2], rotation=270, labelpad=30, fontsize=20)
        cax.ax.tick_params(axis='y', which='major', direction='out', labelsize= 20)
        fig.tight_layout()

    return fig, ax

