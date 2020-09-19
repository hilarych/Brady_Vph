import numpy as np
import scipy.fftpack as sfft
#from copy import copy
import matplotlib.pyplot as plt
from math import pi, atan, e
from scipy.signal import detrend, butter, lfilter, hann, correlate, deconvolve
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
import time


def fk_filter(data_fft2,fst,fsx, vmax=None,vmin=None,filt_downgoing=False,filt_upgoing=False,smooth=5):
    '''
    Given data in the FK domain, filter data with desired velocity range and directions, 
    and return the filtered data in the time domain.
    '''
    # Design filter

    Nx, Nt = data_fft2.shape
    
    # Initialize the filter
    fil = np.ones(shape=data_fft2.shape)
    
    # (Option 1) Filter out certain velocity
    # =============================================================================
    # # Scaling in the f-k domain (Side note: k and w here are actually k/(2pi) and w/(2pi))
    # for k in range(1,int(Nx/2)):
    #     for w in range(1,int(Nt/2)):
    #         v = (w*fst/Nt)/(k*fsx/Nx)
    #         if v > 1800: # and v <50000:
    #             # For the upper left quadrant
    #             fil[k][w]= 0
    #             # For the upper right quadrant
    #             fil[k][-w]= 0
    #             # For the lower left quadrant
    #             fil[-k][w]= 0
    #             # For the lower right quadrant
    #             fil[-k][-w]= 0
    # =============================================================================
    # Scaling in the f-k domain (Side note: k and w here are actually k/(2pi) and w/(2pi))
    
    # (Option 2) Filter out certain velocity + downgoing waves
    for k in range(1,int(Nx/2)):
        for w in range(1,int(Nt/2)):
            v = (w*fst/Nt)/(k*fsx/Nx)
            if filt_downgoing == True:
                # For the lower left quadrant
                fil[-k][w]= 0
                # For the upper right quadrant
                fil[k][-w]= 0
            if filt_upgoing == True:
                # For the upper left quadrant
                fil[k][w]= 0
                # For the lower right quadrant
                fil[-k][-w]= 0
            if vmax!=None and v > vmax:
                # For the upper left quadrant
                fil[k][w]= 0
                # For the upper right quadrant
                fil[k][-w]= 0
                # For the lower left quadrant
                fil[-k][w]= 0
                # For the lower right quadrant
                fil[-k][-w]= 0
            if vmin!=None and v < vmin:
                # For the upper left quadrant
                fil[k][w]= 0
                # For the upper right quadrant
                fil[k][-w]= 0
                # For the lower left quadrant
                fil[-k][w]= 0
                # For the lower right quadrant
                fil[-k][-w]= 0
    
    # Smooth the filter
    fil= gaussian_filter(fil,sigma=smooth,mode='wrap')
    
    # Apply the filter (re-Initialize cct_stacked)
    data_fft_filt = data_fft2*fil
    data_filt= sfft.ifft2(data_fft_filt).real
    
    return data_filt , fil   


def picker(cct_stacked,v_est=[6201,2499],src=180,fst=1000., fsx=1.,shift=0):
    '''
    Pick arrival times for the given cross-correlation time functions (cct).
    
    Parameters
    ----------
    cct_stacked : 2D np.array [z,t]
        Stacked cct.
    v_est : 2-list, optional
        Estimated velocity (m/s) [high,low]. The default is [6201,2499].
    src : int, optional
        Source channel. The default is 180.
    fst : float, optional
        Data sampling rate in time. The default is 1000..
    fsx : float, optional
        Data channel spacing. The default is 1..
    shift : int, optional
        Let shift=len(cct_stacked[0])//2 if input cct has been np.rolled. The default is 0.

    Returns
    -------
    picks : 1D np.array
        Arrival times for each channel.
    t0 : 1D np.array
        Lower bound of the estimated time window for each channel.
    t1 : 1D np.array
        Upper bound of the estimated time window for each channel.
    max_values : 1D np.array
        cct value at the picked time for each channel.

    '''

    v_est=np.asarray(v_est) # Estimated velocity (m/s)
    picks,t0,t1,max_values=[],[],[],[]
    
    for ch, cct in enumerate(cct_stacked):
        dz = abs(ch-src)
        
        # Estimate t window
        t_win = np.rint(dz*fsx/v_est*fst).astype(int) 
        try:
            pick= shift+t_win[0]+np.argmax(cct[shift+t_win[0]:shift+t_win[1]])
            max_value=cct[pick]
            
            # Discard the picks happens at the edges
            if pick == shift+t_win[0] or pick == shift+t_win[1]:
                picks.append(None)
                max_values.append(None)
            else:
                picks.append(pick)
                max_values.append(max_value)
        except ValueError:
            picks.append(None)
            max_values.append(None)       
            
        t0.append(shift+t_win[0])
        t1.append(shift+t_win[1])
    picks=np.asarray(picks)
    max_values=np.asarray(max_values)
    return picks, t0,t1, max_values



def DAS_to_geophone_FD(data,seg, L=10,n=3 ):
    '''
    Given DAS data (in strain rate) of a segment, gauge length, order of FD
    Return the geophone FD like trace (in nanostrain/s)
    
    '''
    pass

def plot_data(data, vm_scale =10,title=None,t_step=5,fs=1000,figsize=(6,3),xlim=None, ylim=None, **kwargs):
    '''
    Plot the data.
    vm_scale: Restrict the maximum value of the color bar to see smaller values. vm_scale =1 : Original scale. Increase vm_scale to see smaller values.
    t_step: Time tick step on the x axis.
    fs: Sampling rate.
    
    '''
    vm = np.percentile(data, 99)
    print("The 99th percentile is {:f}; the max amplitude is {:f}".format(vm, data.max()))
    vm/= vm_scale
    fig, ax = plt.subplots(figsize=(6,3))
    plt.imshow(data, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto',**kwargs)
    plt.colorbar()
    plt.ylabel('Channel')
    plt.xlabel('Elapsed Time (s)')
    plt.title(title)
    
    Nx,Nt=data.shape
    ticks = np.arange(0,Nt,fs*t_step)
    ticklabels=np.arange(0,Nt,fs*t_step)/fs
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels,rotation=90)

        
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    
def plot_FFT(data, vm_scale =10,title=None,t_step=5,fst=1000, fsy=1,figsize=(6,3),xlim=None, ylim=None,**kwargs):
    '''
    Calculate FFT of data and plot.
    '''
    # data_fft2 has zero at the upper left corner
    data_fft2 = sfft.fft2(data)
    # data_fft2_shift has zero at the center
    data_fft2_shift = sfft.fftshift(data_fft2)
    
    # Plot FFT
    vm = np.percentile(np.abs(data_fft2_shift), 99)
    print("The 99th percentile is {:f}; the max amplitude is {:f}".format(vm, data_fft2_shift.max()))
    vm/=vm_scale
    plt.subplots(figsize=figsize)
    plt.imshow(np.abs(data_fft2_shift), cmap="RdBu",  vmin=-vm, vmax=vm, aspect='auto',
              extent=[-fst/2,fst/2,fsy/2,-fsy/2],**kwargs)
    plt.colorbar()
    plt.ylabel(r'$f_x$ (/m)')
    plt.xlabel(r'$f_t$ (Hz)')
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()    

def concatenate_in_time(st_list):
    '''
    Given a list of stream.
    Returns a 2D np.array with all streams concatenated along the last axis (ie. time, axis=-1).
    '''
    # Convert to numpy arrays (x,t)
    data_stack = []
    for st in st_list:
        data_stack.append(np.stack([t.data for t in st.traces]))
    # Concatenate 2 .sgy files 
    data = np.concatenate(data_stack, axis=1)
    return data


def add_distance_stream(stream):
    '''
    Add distance to stream assuming channel spacing = 1 m
    Required file operateion for plotting with obspy.
    '''
    i=0
    for tr in stream:
        tr.stats.distance=i
        i += 1
    return stream
    


def get_t0_DAS(stream):
    '''
    Returns th UTC time of the first time sample of the DAS data.
    Need this function because the starttime in stats in the .sgy file are incorrect.
    Convert the text in the stream header to UTC time.
    '''
    from time import strptime
    from obspy.core import UTCDateTime
    
    UTC_t0 = stream.stats['textual_file_header'].split(b'UTC Timestamp of first sample: ')[1][:30].decode("utf-8") 
    t = strptime(UTC_t0[:-10],'%d-%b-%Y %H:%M:%S')
    t0_DAS = UTCDateTime(t[0],t[1],t[2],t[3],t[4],t[5],int(UTC_t0[-9:-3]))
    
    return t0_DAS
    


def shot_time(meta_vb, stage = 4,VP = 'T122',mode='P1'):
    '''
    Returns the UTC time of the shot.
    '''
    from functools import reduce # For np.intersect1d with more than 2 arrays
    from obspy.core import UTCDateTime
    
    stages = np.squeeze(meta_vb['S']['Stage'][0,0])
    modes= np.squeeze(meta_vb['S']['Mode_sweepNum'][0,0])
    
    VPs = np.squeeze(meta_vb['S']['VP'][0,0])
    inds = reduce(np.intersect1d, (np.where(stages==[stage])[0],
                                   np.where(VPs==[VP])[0],
                                   np.where(modes==[mode])[0]))
    DAQ_times = np.squeeze(meta_vb['S']['DAQ_DateTime'][0,0])
    t= UTCDateTime(DAQ_times[inds][0][0][:14])
    
    return t

def FTP_download(files,local_dir,remote_dir,server='ftp://roftp.ssec.wisc.edu/',
                 username='hilarych',password='hilarych@mit.edu'):
    '''
    Download file from the wisc FTP site.
    files: A list of file names
    '''
    from ftplib import FTP
    from datetime import datetime

    ftp = FTP(server)
    ftp.login(username,password)
    
    # Print out the files
    for file in files:
    	print("Downloading..." + file)
    	ftp.retrbinary("RETR " + file ,open(local_dir + file, 'wb').write)
    ftp.close()
    
    
    
def find_vb_sgy(meta_vb,data_dir, stage = 4,VP = 'T122',mode='P1'):
    '''
    Find the vibroseis .sgy files given the stage and shot point (VP).
    Find the file by 
    1. Getting the actual time t of the shot from meta_vb and define.
    2. Define the time range t_range of the data starttime/endtime based on t.
    2. Load .sgy file in data_dir that has starttime within t_range (tells the time by the filenames).
    
    '''
    import os
    from glob import glob
    
    t_range=[]  # Time trange of the [starttime, endtime] of the data
    #for shot in range(1):
    t= shot_time(meta_vb=meta_vb,stage=stage,VP=VP,mode=mode) # Actual shot time of the mode

    t_range.append(t-33) # Include a tolerance of 33 sec (the data is 30 sec long in time, add another 3 sec for tapering... etc)
    t_range.append(t+33)
    # Note: This 3 sec tolerance corresponds to t0_data (the allowance time in the data before the actual shot time in Vibroseis.py 
        
    directory = data_dir+"{}".format(t_range[0].strftime('%Y%m%d'))
    file_list = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in sorted([f for f in filenames if f.endswith(".sgy")]):
            basename, ext = os.path.splitext(filename)
            try:
                number = int('20'+basename[-12:])
            except ValueError:
                continue  # not numeric
            # Allow 5 sec before the first shot and 30 sec after the last shot (ST3)
            #if ((int(tt[0][:14])-5) <= number <= (30+int(tt[-1][:14]))):
            if ( int(t_range[0].strftime('%Y%m%d%H%M%S')) <= number <= int(t_range[1].strftime('%Y%m%d%H%M%S'))):
                # process file
                file_list.append(os.path.join(directory, filename))
    print(file_list)            
            
    return file_list

def RMS(signal):
    '''
    Function that Calculate Root Mean Square
    '''
    return np.sqrt(np.mean(signal**2))




# a=np.array([[1,2,3],[4,5,6]])
# tr_energy = np.sum(a**2,axis=1)
# tr_weights = np.sum(tr_energy)/tr_energy
# tr_weights /= np.sum(tr_weights) 
# tr_weights
# data_weighted = (a.T@ np.diag(tr_weights)).T 
# data_weighted
# common_signals =  np.sum(data_weighted, axis = 0)
# common_signals

def est_common_signals(data):
    '''
    Estimate the common signal in the data related to the interrogator shake.
    The estimation is based on energy reweighted data. 
    Note: Weights = Normalized (total energy)/(trace energy)
    '''
    # Procedure for getting the common signals related to the interrogator shake
    tr_energy = np.sum(data**2,axis=1)       # The energy in each trace
    tr_weights = np.sum(tr_energy)/tr_energy # The weights for each trace
    tr_weights /= np.sum(tr_weights)         # Normalize the weights
    data_weighted = (data.T@ np.diag(tr_weights)).T  # Apply the weightings
    # Stack all traces to produce one summary trace (the common signal) 
    common_signals =  np.sum(data_weighted, axis = 0)
    return common_signals
    
def fk_scaling(data,fst=1000,fsx = 1,regulize=0.1,output='f'):
    '''
    Given 2D data i (x,t), return fk-scaled data.
    Purpose: Convert unit from fiber strain to particle velocity.
    regulize: Small amount to put in the scale in special cases k=0 and w=0
    output='f': return data in the f-domain; ='t': in the t-doamin
    
    Brady data:
    fst = 1000 # sample/s
    fsx = 1 # sample/m
    Nt =30000 (one file)
    Nx=384
    
    The indexing is for even numbers of Nx and Nt. See:
    https://docs.scipy.org/doc/scipy-1.1.0/reference/tutorial/fftpack.html
    '''
    Nx, Nt = data.shape
    
    # Calculate fft2D (not shifted) 
    data_fft2 = sfft.fft2(data)
    # Initialize the scaled data
    data_fft2_fkscaled = data_fft2.copy()
    # Scaling in the f-k domain (Side note: k and w here are actually k/(2pi) and w/(2pi))
    for k in range(1,int(Nx/2)):
        for w in range(1,int(Nt/2)):
            scale = (w*fst/Nt)/(k*fsx/Nx)
            # For the upper left quadrant
            data_fft2_fkscaled[k][w]= data_fft2[k][w]* scale
            # For the upper right quadrant
            data_fft2_fkscaled[k][-w]= data_fft2[k][-w]* -scale
            # For the lower left quadrant
            data_fft2_fkscaled[-k][w]= data_fft2[-k][w]* -scale
            # For the lower right quadrant
            data_fft2_fkscaled[-k][-w]= data_fft2[-k][-w]* scale

    # Deal with special cases: k=0 and w=0
    for k in range(1,int(Nx/2)):
        w = regulize  # w \approx 0
        scale =  (w*fst/Nt)/(k*fsx/Nx)
        data_fft2_fkscaled[k][0]= data_fft2[k][0]* scale
        data_fft2_fkscaled[-k][0]= data_fft2[-k][0]* -scale
        
    for w in range(1,int(Nt/2)):
        k = regulize  # k \approx 0
        scale = (w*fst/Nt)/(k*fsx/Nx)
        data_fft2_fkscaled[0][w]= data_fft2[0][w]* scale
        data_fft2_fkscaled[0][-w]= data_fft2[0][-w]* -scale
    
    # Deal with the origin
    scale =  (fst/Nt)/(fsx/Nx)
    data_fft2_fkscaled[0][0]= data_fft2[0][0]* scale
    
    if output == 't' or output =='T':
        # Inverse fft
        data_fkscaled = sfft.ifft2(data_fft2_fkscaled).real
        return data_fkscaled
    elif output == 'f' or output =='F':
        return data_fft2_fkscaled
        
def k_scaling(data,fsx = 1,regulize=0.1,output='f'):
    '''
    Given 2D data i (x,t), return fk-scaled data.
    Purpose: Convert unit from fiber strain to particle velocity.
    regulize: Small amount to put in the scale in special cases k=0 and w=0
    output='f': return data in the f-domain; ='t': in the t-doamin
    
    Brady data:
    fst = 1000 # sample/s
    fsx = 1 # sample/m
    Nt =30000 (one file)
    Nx=384
    
    The indexing is for even numbers of Nx and Nt. See:
    https://docs.scipy.org/doc/scipy-1.1.0/reference/tutorial/fftpack.html
    '''
    Nx, Nt = data.shape
    
    # Calculate fft2D (not shifted) 
    data_fft2 = sfft.fft2(data)
    # Initialize the scaled data
    data_fft2_kscaled = data_fft2.copy()
    # Scaling in the f-k domain (Side note: k and w here are actually k/(2pi) and w/(2pi))
    for k in range(1,int(Nx/2)):
        scale = 1/(1j*k*fsx/Nx)
        # For the upper half
        data_fft2_kscaled[k]= data_fft2[k]* scale
        # For the lower half
        data_fft2_kscaled[-k]= data_fft2[-k]* -scale

    # Deal with special cases: k=0 
    k = regulize  # k \approx 0
    scale = 1/(1j*k*fsx/Nx)
    data_fft2_kscaled[0]= data_fft2[0]* scale
    
    if output == 't' or output =='T':
        # Inverse fft
        data_kscaled = sfft.ifft2(data_fft2_kscaled).real
        return data_kscaled
    elif output == 'f' or output =='F':
        return data_fft2_kscaled
    
def k_scaling2(data,fsx = 1,regulize=0.1,output='f'):
    '''
    Given 2D data i (x,t), return fk-scaled data.
    Purpose: Convert unit from fiber strain to particle velocity.
    regulize: Small amount to put in the scale in special cases k=0 and w=0
    output='f': return data in the f-domain; ='t': in the t-doamin
    
    Brady data:
    fst = 1000 # sample/s
    fsx = 1 # sample/m
    Nt =30000 (one file)
    Nx=384
    
    The indexing is for even numbers of Nx and Nt. See:
    https://docs.scipy.org/doc/scipy-1.1.0/reference/tutorial/fftpack.html
    '''
    Nx = len(data)
    # Initialize the scaled data
    fiber_fft_T = np.zeros_like(data.T,dtype=complex)    
    
    for t,fiber in enumerate(fiber_fft_T):
        # Calculate fft (not shifted) 
        fiber_fft_T[t,:] = sfft.fft(data.T[t])
        
        # Scaling in the k domain (Side note: k here is actually k/(2pi) 
        for k in range(1,int(Nx/2)):
            scale = 1/(1j*k*fsx/Nx)
            fiber_fft_T[t][k]*= scale
            fiber_fft_T[t][-k]*= -scale

        # Deal with special cases: k=0 
        k = regulize  # k \approx 0
        scale = 1/(1j*k*fsx/Nx)
        fiber_fft_T[t][0]*= scale
    
    if output == 't' or output =='T':
        for t, fiber in enumerate(fiber_fft_T):
            # Inverse fft
            fiber_fft_T[t]= sfft.ifft(fiber).real
        return fiber_fft_T.T
    elif output == 'f' or output =='F':
        return fiber_fft_T.T
# data=data_iseg
# fsx=fsx_DAS
# regulize=0.1
# output='t'
# Nx = len(data)
# # Initialize the scaled data
# fiber_fft_T = np.zeros_like(data.T,dtype=complex)    
# t=0
# # for t,fiber in enumerate(fiber_fft_T):
# # Calculate fft (not shifted) 
# fiber_fft_T[t,:] = sfft.fft(fiber_fft_T[t])

# # Scaling in the k domain (Side note: k here is actually k/(2pi) 
# for k in range(1,int(Nx/2)):
#     scale = 1/(1j*k*fsx/Nx)
#     fiber_fft_T[t][k]*= scale
#     fiber_fft_T[t][-k]*= -scale

# # Deal with special cases: k=0 
# k = regulize  # k \approx 0
# scale = 1/(1j*k*fsx/Nx)
# fiber_fft_T[t][0]*= scale

# if output == 't' or output =='T':
#     for t, fiber in enumerate(fiber_fft_T):
#         # Inverse fft
#         fiber_fft_T[t]= sfft.ifft(fiber).real
#     return fiber_fft_T.T
# elif output == 'f' or output =='F':
#     return fiber_fft_T.T

def k_scaling_sym(data,fsx = 1,regulize=0.1,output='f',flipflag=1,taper=None):
    '''
    Given 2D data i (x,t), return fk-scaled data.
    Purpose: Convert unit from fiber strain to particle velocity.
    regulize: Small amount to put in the scale in special cases k=0 and w=0
    output='f': return data in the f-domain; ='t': in the t-doamin
    flipflag: =1: Put the flipped data before the original data. 
    
    Brady data:
    fst = 1000 # sample/s
    fsx = 1 # sample/m
    Nt =30000 (one file)
    Nx=384
    
    The indexing is for even numbers of Nx and Nt. See:
    https://docs.scipy.org/doc/scipy-1.1.0/reference/tutorial/fftpack.html
    '''
    # Nx = len(data)
    
    # Flip the data along z and stack it with the original (z,t) data.
    if flipflag ==1:
        data_sym= np.vstack( (np.flip(data, axis=0), data) )
    elif flipflag ==0:
        data_sym= np.vstack( (data , np.flip(data, axis=0)) )
        
    Nx = len(data_sym)
    print('Sample number in x for the symmetrical signal = ',Nx)
    # Initialize the scaled data
    fiber_fft_T = np.zeros_like(data_sym.T,dtype=complex)    
    
    for t,fiber in enumerate(fiber_fft_T):
        # Taper
        if taper != None:
            # Calculate fft (not shifted) 
            fiber_fft_T[t,:] = sfft.fft(hann_taper(data_sym.T[t],taper))
        elif taper == None:
            fiber_fft_T[t,:] = sfft.fft(data_sym.T[t])
        # Scaling in the k domain (Side note: k here is actually k/(2pi) 
        for k in range(1,int(Nx/2)):
            scale = 1/(1j*k*fsx/Nx)
            fiber_fft_T[t][k]*= scale
            fiber_fft_T[t][-k]*= -scale

        # Deal with special cases: k=0 
        k = regulize  # k \approx 0
        scale = 1/(1j*k*fsx/Nx)
        fiber_fft_T[t][0]*= scale
    
    if output == 't' or output =='T':
        for t, fiber in enumerate(fiber_fft_T):
            # Inverse fft
            fiber_fft_T[t]= sfft.ifft(fiber).real
        return fiber_fft_T[:,int(Nx//2):].T
    elif output == 'f' or output =='F':
        return fiber_fft_T.T


def hann_taper(data,percentage=0.1,wlen=None,left_right='both'):
    '''
    Design a Hann taper that tapers the first and last wlen samples.
    Default: Taper length based on percentage of the total points
            If wlen != None, use length as the taper length.
    
    '''
    npts = np.size(data,-1)  # Default: Apply taper to the last dimension
    
    if wlen == None:
        wlen = int(round(npts*percentage))
    
    window = hann(wlen*2)
    #window = scipy.signal.hann(int(0.05 * npts))
    if left_right=='both':
        left = window[:wlen]
        right = window[wlen:]
    elif left_right=='left':
        left = window[:wlen]
        right = np.ones(wlen)
    elif left_right=='right':
        left = np.ones(wlen)
        right = window[wlen:]
    else:
        raise Exception('Available options: "both", "left", "right"')
        
    middle = np.ones(int(npts-wlen*2))
    window = np.concatenate((left, middle, right))
    data_tapered = data* window
    return data_tapered



def response_filter(N,  f0 = 5.,damp=0.7,fs=500,gain_in_dB=38,gain_limit=10):
    '''
    Cf. Shearer - Introduction to sSeismology p 323 & Zland 3C responce: 
    https://www.passcal.nmt.edu/content/instrumentation/sensors/high-frequency-sensors/nodes
    
    Output Z is the filter in freq domain.

    # You can check the freq mag and phase response using the following code:
    
    from Brady_processing import *
    N=1000000
    f0= 5.
    gain = 10**(38/20)  # Gain: 36 dB matches the chart on the website the most
    Z= response_filter(N=N, f0 = f0,damp=0.7,fs=500)*gain
    f = np.array([n for n in range(N)])*fs_nd/N 
    plt.loglog(f,abs(Z))
    # Draw a vertical line at corner frequency
    plt.axvline(x=f0, linestyle='dashed', color='grey', linewidth=1)
    # Check the magnitude decay at the cutoff frequency f0 (the point where mag drop -3dB):
    if0 = int(round(f0/fs_nd*N)) # index of the f0
    dB_at_f0 = 20*np.log10(abs(Z[if0])/max(abs(Z)))
    print('At f0, magnitude drops to:{} dB (should be close to -3)'.format(dB_at_f0))
    # Draw a horizontal line at mag=80 to check the maximum Z (should line up with the flat end)
    plt.axhline(y=80, linestyle='dotted', color='tab:red', linewidth=1) 
    plt.axhline(y=70, linestyle='dotted', color='tab:red', linewidth=1)
    plt.axhline(y=60, linestyle='dotted', color='tab:red', linewidth=1)
    print('Max(Z)={} (should be close to 80)'.format(max(abs(Z))))
    plt.xlabel('Frequency (Hz)')
    plt.yticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
    plt.tight_layout()
    #(Check) Response: Phase
    plt.semilogx(f,-np.arctan(Z.imag/Z.real)/pi*180 )
    plt.xlabel('Frequency (Hz)')
    # Note that x axis is n (ex: for fs=500,N=45000, fs/N=90 is where 1Hz is)
    # https://www.passcal.nmt.edu/content/instrumentation/sensors/high-frequency-sensors/nodes
    
    Note: Although the nodal geophone measures velocity, the displacement relationship in Shearer(11.12) matches the the responses given by the manufacturer more.
    '''

    omega0 = 2*pi*f0 # Resonance frequency
    # 2 eps = D/m 
    # k/m = omega0^2 
    eps = damp*omega0 # Shearer eq 11.4-5; text under Fig 11.2
    omega = np.array([2*pi*n for n in range(N)])*fs/N # angular frequency
    
    num = omega**2 # Shearer 11.12 numerator (for displacement)
    #num = (-1j*omega) # Shearer 11.12 numerator (for velocity)
    den = omega0**2-2*eps*1j*omega-omega**2 # Shearer 11.12 denominator
    
    Z=num/den #Shearer eq 11.12
    
    gain = 10**(gain_in_dB/20)  # Gain: 38 dB matches the chart on the website the most
    Z*= gain
    
    if gain_limit !=0:
        Zm = abs(Z)
        Za = np.angle(Z)
        np.clip(Zm, 1/gain_limit, None,out =Zm) # Hm(Hm<.1)=1/gainlimit;
        Z= Zm*e**(1j*Za)
    
    return Z

def remove_response(data,f0 = 5.,damp=0.7,fs=500, gain_in_dB=38,gain_limit=10):
    '''
    Given data in the t-domain and instrument responce parameters:
        f0: Resonance frequency (Hz)
        damp: Damping constant
        fs: Sampling rate (sample/s)
    '''
   
    N = int(data.shape[-1])
    Z = response_filter(N=N,f0=f0,damp=damp,fs=fs,gain_in_dB=gain_in_dB,gain_limit=gain_limit)
    Data = sfft.fft(data)
    
    Data /= Z 
    output = sfft.ifft(Data)
        
 
    return output
        
    
def nodal_pz_compensation(signal,fs,resfreq=5.,damp=0.7,gainlimit=0,plot_freqres=False):
    '''
    Pole-Zero Function
    
    pole zero compensation., based on information about poles and zeros from the
    manufacturer.
     
    signal       : Input Signal. Should be a column vector
    fs           : Sampling Frequency (Hz)
    resfreq      : Resonant Frequency (Hz)
    damp         : Damping Factor
    gainlimit    : Gain Limit. For no gain limit, set = 0.
    plot_freqres : Create plots for Pole-Zero Compensation  (True/False)
    '''
    # Determine the total number of samples
    N = len(signal)

    # define frequency rangen(rads/sec)
    w = 2*pi*np.arange(0,fs/2,fs/N) # w = 2*pi*(0:fs/N:fs/2)

    # radius of pole zero circle and plot on pole-zero unit circle plot
    wo = 2*pi*resfreq

    # Transfer function input
    s = 1j*w # s = sqrt(-1).*w
    
    # calculating the pole from known quantities
    # p(1) = -damp.*wo + wo.*sqrt(-1).*sqrt(1-damp.^2);
    # p(2) = -damp.*wo - wo.*sqrt(-1).*sqrt(1-damp.^2);
    p = np.array([-damp*wo + wo*1j*np.sqrt(1-damp**2), -damp*wo - wo*1j*np.sqrt(1-damp**2)])
    # assume zeros are at zero
    z = np.array([0+0j, 0+0j])
    
    # =============================================================================
    #     if plot_freqres == True:
    #         #plt.figure()
    #         #circle=Circle((0,0),radius=wo,linestyle='--',edgecolor='tab:red',linewidth=1)
    #         #plt.title('Pole-Zero Plot')
    #         # create pole-zero unit circle plot
    #         zplane(z,p)
    # =============================================================================
    #gain = 10**(38/20)  # Gain: 38 dB matches the chart on the website the most
    
    # Calculation of the Transfer Function
    T = ((s-z[0])*(s-z[1]))  /  ((s-p[0])*(s-p[1])) #*gain
 
    # Magnitude Response:  
    Hm= abs(T)
    # Unconstrained Magnitude Response
    Hm2 = Hm.copy()

    # Constrained Magnitude Response
    if gainlimit != 0:
        np.putmask(Hm, Hm<1/gainlimit,values=1/gainlimit) # Hm(Hm<.1)=1/gainlimit;
 
    # Back Calculation of Transfer Function including forced constraints (Previously System Response)
    H = Hm*e**(1j*np.angle(T))
    #H = Hm*e**(1j*np.arctan2(T.imag,T.real))

    # Phase Response
    #Ha = np.arctan2(T.imag,T.real)/pi/2

    if plot_freqres == True:
        # Magnitude Respone Plot
        plt.figure()
        plt.loglog(w/2/pi,Hm2, label='Pure Magnitude',alpha=0.8) 
        plt.loglog(w/2/pi,Hm, label='Constrained')
        #plt.xlim(0.1,20)
        #plt.ylim(0.0001,12)
        plt.title('Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        #plt.axhline(y=80, linestyle='dotted', color='tab:red', linewidth=1) 
        # Phase Response Plot
        plt.figure()
        plt.semilogx(w/2/pi,np.angle(H)*180/np.pi)
        #plt.xlim(0.1, 20)
        #plt.ylim(-0.5, 0.5)
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degree)')
        plt.tight_layout()
        
        if0 = int(round(resfreq/fs*N)) # index of the f0
        dB_at_f0 = 20*np.log10(Hm[if0]/max(Hm))
        print('Max(Z)={} (should be close to 80)'.format(max(Hm)))
        print('At f0, magnitude drops to:{} dB (should be close to -3)'.format(dB_at_f0))
    
    # 'H' can then be applied to the signal via the following block of code
    # (signal is your vector from your receiver):
    # 
    # Preform a Fast Fourier Transform
    signalfft = sfft.fft(signal)

    # Select only positive Frequencies
    signalfft = signalfft[0:int(N/2)] # Even N  # signalfft = signalfft(1:numel(signal)/2+1) %odd number
    #signalfft = signalfft(1:11001);   #even number

    # Apply pole/zero compensation
    signalfft = signalfft/H.conj() #signalfft = (signalfft)./H';

    signalfft = np.concatenate((signalfft, signalfft[::-1][:-1]))
    # Inverse Fast Fourier Transform
    newsignal = sfft.ifft(signalfft,N)
    return newsignal

