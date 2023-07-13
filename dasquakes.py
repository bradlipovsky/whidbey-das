import numpy as np
from tqdm import tqdm
import pickle
from time import perf_counter
from scipy.sparse.linalg import svds
import datetime
import h5py
import glob
from scipy.signal import detrend, butter, filtfilt
from numpy.fft import fftshift, fft2, fftfreq
from datetime import datetime as DT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def data_wrangler(cable,record_length,t0):
    if cable == 'seadasn':
        prefix = 'seadasn'
        network_name = 'SeaDAS-N'
        if t0 < datetime.datetime(2022, 6, 20, 0, 0, 0):
            datastore='/data/data0/seadasn_2022-03-17_2022-06-20/'
        elif (t0 >= datetime.datetime(2022, 6, 20, 0, 0, 0)) and (t0 < datetime.datetime(2022, 10, 7, 0, 0, 0)):
            datastore='/data/data7/seadasn_2022-06-21_2022-10-06/'
        elif (t0 >= datetime.datetime(2022, 10, 7, 0, 0, 0)) and (t0 < datetime.datetime(2023, 1, 14, 0, 0, 0)):
            datastore='/data/data3/seadasn_2022-10-07_2023-01-13/'
        else:
            datastore='/data/data4/seadasn/'

    elif cable == 'whidbey':
        prefix = 'whidbey'
        network_name='Whidbey-DAS'
        if t0 < datetime.datetime(2022,10,23,4,50,0):
            datastore = '/data/data5/Converted/'
        else:
            datastore = '/data/data6/whidbey/'
        
    return prefix, network_name, datastore


def dt_to_utc_format(t):
    from obspy import UTCDateTime
    return UTCDateTime(t.strftime('%Y-%m-%dT%H:%M:%S'))

def utc_to_dt_format(t):
    dt_str = t.strftime('%Y/%m/%d %H:%M:%S')
    format1  = "%Y/%m/%d %H:%M:%S"
    dt_utc = DT.strptime(dt_str, format1)
    return dt_utc
    
def sintela_to_datetime(sintela_times):
    '''
    returns an array of datetime.datetime 
    ''' 
    days1970 = datetime.date(1970, 1, 1).toordinal()

    # Vectorize everything
    converttime = np.vectorize(datetime.datetime.fromordinal)
    addday_lambda = lambda x : datetime.timedelta(days=x)
    adddays = np.vectorize(addday_lambda )
    
    day = days1970 + sintela_times/1e6/60/60/24
    thisDateTime = converttime(np.floor(day).astype(int))
    dayFraction = day-np.floor(day)
    thisDateTime = thisDateTime + adddays(dayFraction)

    return thisDateTime

def open_sintela_file(file_base_name,t0,pth,
                      chan_min=0,
                      chan_max=-1,
                      number_of_files=1,
                      verbose=False,
                      pad=False):

    data = np.array([])
    time = np.array([])

    
    dt = datetime.timedelta(minutes=1) # Assume one minute file duration
    this_files_date = t0
    
    for i in range(number_of_files):
        
        # Construct the "date string" part of the filename
        date_str = this_files_date.strftime("%Y-%m-%d_%H-%M")
    
        # Construct the PARTIAL file name (path and name, but no second or filenumber):
#         this_file = f'{pth}{file_base_name}_{date_str}_UTC_{file_number:06}.h5'
        partial_file_name = f'{pth}{file_base_name}_{date_str}'
        file_search = glob.glob(f'{partial_file_name}*h5')
        if verbose:
            print(f'Searching for files matching: {partial_file_name}*h5')
        if len(file_search) > 1:
            raise ValueError("Why are there more than one files? That shouldn't be possible!")
        elif len(file_search) == 0:
            raise ValueError("Why are there ZERO files? That shouldn't be possible!")
        else:
            this_file = file_search[0]
        
        try:
            f = h5py.File(this_file,'r')
            this_data = np.array(
                f['Acquisition/Raw[0]/RawData'][:,chan_min:chan_max])
            this_time = np.array(
                f['Acquisition/Raw[0]/RawDataTime'])
            
            if i == 0:
                time = sintela_to_datetime(this_time)
                data = this_data
                attrs=dict(f['Acquisition'].attrs)
            else:
                data = np.concatenate((data, this_data ))
                time = np.concatenate((time, this_time ))
                
        except Exception as e: 
            print('File problem with: %s'%this_file)
            print(e)
            
            # There's probably a better way to handle this...
            #             return [-1], [-1], [-1]


        this_files_date = this_files_date + dt
    
    #if pad==True:
        # Add columns of zeros to give data matrix the correct dimensions
        
    return data, time, attrs

def local_earthquake_quicklook(dates,datafilt,st,st2,
                        x_max,stitle,filename=None,
                        skip_seismograms=False,
                        das_vmax=0.1,
                        network_name=''):
    '''
    Make a nice plot of the DAS data and some local seismic stations
    '''
    dx = x_max / datafilt.shape[1]
    fig,ax=plt.subplots(figsize=(8,12))
    date_format = mdates.DateFormatter('%H:%M:%S')
    
    # Subplot: DAS Data
    ax=plt.subplot(4,1,1)
    ax.set_title(f'{network_name}')
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(datafilt.T,vmin=-das_vmax,vmax=das_vmax,
               cmap='seismic',aspect='auto', 
               extent=[x_lims[0],x_lims[-1],x_max,0])
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()
    
    # Subplot: Single DAS Channel
    ax = plt.subplot(4,1,2)
    fig.patch.set_facecolor('w')
#     graph_spacing = -400
    graph_spacing = -20
    for jj in (41,400,800,1400):
        plt.plot(dates,datafilt[:,jj]-jj/graph_spacing,label=f'OD = {int(jj*dx)} m')
    plt.legend(loc='upper right')
    ax.set_title(f'{network_name} Individual Channels')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.grid()


    if skip_seismograms==False:
        
        # Subplot:  station 1
        ax = plt.subplot(4,1,3)
        for tr in st:
            times_from_das = np.linspace(x_lims[0],x_lims[-1],len(tr.data))
            plt.plot(times_from_das,tr.data)
        fig.patch.set_facecolor('w')
        ax.set_title('UW NOWS HNN')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das),max(times_from_das)))
        plt.grid()
    

        # Subplot:  station 2
        ax = plt.subplot(4,1,4)
        for tr in st2:
            times_from_das = np.linspace(x_lims[0],x_lims[-1],len(tr.data))
            plt.plot(times_from_das,tr.data)
        fig.patch.set_facecolor('w')
        ax.set_title('IU COR BH1')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das),max(times_from_das)))
        plt.grid()
    
    

    fig.suptitle(stitle,fontsize=20)
    plt.tight_layout()
    
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    
    
def data_quicklook(     dates,datafilt,
                        x_max,stitle,filename=None,
                        das_vmax=0.1,
                        network_name='',
                        ylim=None):
    '''
    Make a nice plot of DAS data 
    '''
    dx = x_max / datafilt.shape[1]
    fig,ax=plt.subplots(figsize=(10,10))
    date_format = mdates.DateFormatter('%H:%M:%S')
    
    # Subplot: DAS Data

    ax.set_title(f'{network_name}')
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(datafilt.T,vmin=-das_vmax,vmax=das_vmax,
               cmap='seismic',aspect='auto', 
               extent=[x_lims[0],x_lims[-1],x_max,0])
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
    

    fig.suptitle(stitle,fontsize=20)
    plt.tight_layout()
    
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
def fk_analysis(t0, 
                draw_figure = True,
                fs=10, dx=6.38,
                cable = 'whidbey', 
                record_length = 1,
                distance_range=[7816,10208],
                verbose = True,
                anti_alias = True):
    '''
    This function takes inputs that describe a subset of a DAS deployment and returns FK data.

    The default distance range represents the subsea part of the whidbey cable
    
    Note the subtlty in the definition of dx_input that dx quantities always need at least two decimal places. 
    I should code this up in a smarter way.  So for example you want to use 3.19 not 3.2.

    '''
    
    prefix, network_name, datastore = data_wrangler(cable,record_length,t0)
    try:
        data,dates,attrs = open_sintela_file(prefix,
                                         t0,
                                         datastore,
                                         number_of_files=record_length,
                                         verbose=True)
    except:
        print("Failed to load data for specified time range")
        return [np.nan], [np.nan], [np.nan]
    
    ''' 
    Downsampling
    '''
    fs_input = 2*attrs['MaximumFrequency']
    t_downsample_factor = int(fs_input/fs) # example: fs_input=20, fs=10, then factor = 20/10 = 2
    if t_downsample_factor == 0:
        print("ERROR: Desired fs  >  input fs.")
        return [np.nan], [np.nan], [np.nan]
    if verbose: print(f"Input fs = {fs_input}, desired fs = {fs}")
    if verbose: print(f"Temporal downsampling factor = {t_downsample_factor}")

    if anti_alias:
        b,a = butter(4,fs/2,'lowpass',fs=fs_input)
        data_filt = filtfilt(b,a,data,axis=0)
    else:
        data_filt = data
    
    dx_input = round(attrs['SpatialSamplingInterval'],2)
    x_downsample_factor = int(dx/dx_input) # example: if dx_input=5 and dx=10 then factor=10/5 = 2
    if x_downsample_factor == 0:
        print("ERROR: Desired dx  <  input dx.")
        return [np.nan], [np.nan], [np.nan]
    if verbose: print(f"Input dx = {dx_input}, desired dx = {dx}")
    if verbose: print(f"Spatial downsampling factor = {x_downsample_factor}")
    
    x1 = int(distance_range[0]/dx_input)
    x2 = int(distance_range[1]/dx_input)

    subsea_data = detrend(data_filt[:,x1:x2:x_downsample_factor])
    downsampled_subsea_data = subsea_data[::t_downsample_factor,:]
    if verbose: print(f"Data dimension: {downsampled_subsea_data.shape}")
    
    '''
    Calculate FFT
    '''

    ft = fftshift(fft2(downsampled_subsea_data))
    f = fftshift(fftfreq(downsampled_subsea_data.shape[0], d=1/fs))
    k = fftshift(fftfreq(downsampled_subsea_data.shape[1], d=dx))
    
    return ft,f,k
    
    
    
    
    
    
def svd_analysis(N=24,dt=60,dx=6.38,fs=10,
                 distance_range=[7816,10208],
                 record_length=2,
                 start_time = datetime.datetime(2022, 5, 8, 0, 0, 0), 
                 outputfile='svd.pickle',
                 f_bandstop = None,
                 k_bandstop = None,
                 verbose=False):
    '''
    Performs the entire FK-domain SVD analysis of a collection of DAS data files.
    
    INPUTS:
    
        fs     Sampling rate used within each fk plot, Hz
        N      Number of fk plots to make
        dt     Sampling rate between the fk plots, minutes    
        
    
    Notes:
        
    The f_bandstop should remove entries from the data matrix rather than just zero'ing them out.
    This would save space and I just implemented the zero'ing approach because it's fast.
    
    fk filtering is setup as a list of filter bandstops, ie
        f_bandstop = [[f1,f2],[f3,f4]]
        k_bandstop = [[k1,k2],[k3,k4]]
    
    '''
    
    '''
    Build the data matrix
    '''
    file_duration = 60 #seconds.  this shouldn't change through the deployment.

    # Number of time steps in each sample
    nt = int(record_length*file_duration*fs) 
    
    # Number of subsea channels at Whidbey. This also changes.
    nx = int((distance_range[1]-distance_range[0]+1)/dx) 
    
    if verbose: print(f"nx={nx}, nt={nt}")
    
    D = np.zeros((nx*nt,N))
    t = []

    for i in tqdm(range(N)):

        this_time = start_time + i*datetime.timedelta(minutes=dt)
        t.append(this_time)
        ft,f,k = fk_analysis(this_time,draw_figure=False,fs=fs, 
                            record_length = record_length,
                            distance_range = distance_range,
                            verbose=verbose)
        
        
        if len(ft) == 1:
            continue
            
        if f_bandstop is not None:
            KK,FF = np.meshgrid(k,f)
            for fstop,kstop in zip(f_bandstop,k_bandstop):
                zero_these = (FF>fstop[0]) & (FF<fstop[1]) & (KK>kstop[0]) & (KK<kstop[1])
                ft[zero_these] = 0

        this_nt = ft.shape[0]
        this_nx = ft.shape[1]

        if this_nt < nt:
            ft_new = np.zeros((nt,nx))
            ft_new[0:this_nt,0:nx] = np.abs(ft)
            this_column =  ft_new.flatten()
        elif this_nt > nt:
            ft_new = np.zeros((nt,nx))
            ft_new[0:nt,0:nx] = np.abs(ft[0:nt,0:nx])
            this_column =  ft_new.flatten()
        else:
            # This should be the typically situation:  
            #    there are exactly as many time steps as we anticipated.
            this_column = np.abs( ft.flatten() )

        D[:,i] = this_column


    t=np.array(t)
    
    
    '''
    Calculate the SVD
    '''
    ns = N
    t1 = perf_counter()
    U,S,V = svds( D[:,0:ns] )
    t = t[0:ns]
    print(f'SVD runtime:   {perf_counter()-t1} s')

    
    
    '''
    Recalculate f and k (in case the last column error'ed out)
    '''

    f=fftshift(fftfreq(nt,d=1/fs))
    k=fftshift(fftfreq(nx,d=dx))
    
    # open a file, where you ant to store the data
    file = open(outputfile, 'wb')
    pickle.dump((U,S,V,t,f,k,nt,nx), file)
    file.close()
    
    

def plot_svd(f,k,t,mode,time_series,var,i,vm=0.5,
             time_series_range=None,
             filename=None,
             fig=None,
             ax1=None,
             ax2=None,
             flim=2.5,
             klim=0.04):
#     import matplotlib 
#     matplotlib.rc('xtick', labelsize=20) 
#     matplotlib.rc('ytick', labelsize=20) 
#     matplotlib.rc('font', family='normal',size=22)
    
    '''
    Plot the results of the FK/SVD analysis
    '''
    
    if (fig is None) and (ax1 is None) and (ax2 is None):
        plt.subplots(2,1,figsize=(10,10))
        ax1=plt.subplot(2,1,1)
        ax2=plt.subplot(2,1,2)
        
    ax1.set_title(f'SV {i} ({var:.2f}% variance)',fontsize=20)
    ax2.set_title(f'SV {i} Time Series',fontsize=20)
    c=ax1.imshow(mode,aspect='auto',vmin=0,vmax=vm,extent=[k[0],k[-1],f[0],f[-1]],cmap='gray_r')

    ax1.set_ylim([-flim,flim])
    ax1.set_xlim([-klim,klim])
    ax1.set_xlabel('Wavenumber (1/m)',fontsize=18)
    ax1.set_ylabel('Frequency (Hz)',fontsize=18)

    ax2.set_ylabel('Normalized Amplitude',fontsize=18)
#     plt.colorbar()

    
    ind = np.where(np.abs(time_series)>1e-10)
#     sign_change = np.sign(np.mean(time_series))
#     ax2.plot(t[ind],time_series[ind]*sign_change,'o')
    ax2.plot(t[ind],time_series[ind],'o',alpha=0.5)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    ax2.grid()
    if time_series_range is not None:
        ax2.set_ylim(time_series_range)
    plt.tight_layout()
    
    if i==2:
        ax2.set_ylim([-0.025,0.04])
                     
    if filename is not None:
        plt.savefig(filename)
#     else:
#         plt.show()