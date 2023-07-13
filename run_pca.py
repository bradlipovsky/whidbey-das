'''
SVD analysis of Distributed Acoustic Sensing (DAS) frequency-wavenumber (FK) plots.

TODO: Add wind speed data
TODO: Convert frequency-wavenumber to frequency-wavespeed
TODO: Experiment with stacking parameters.  Note that SGWs appear with longer stacking windows.
TODO: Speed up the calculations using multiprocessing.
'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
import datetime
from time import perf_counter
from scipy.sparse.linalg import svds
from tqdm import tqdm
from dasquakes import *
import pickle

def main():
    t0 = perf_counter()

    '''
    Carry out the SVD analysis
    
    default parameters:
    svd_analysis(N=24,dt=60,dx=6.38,fs=10,
                     distance_range=[7816,10208], 
                     record_length=2,
                     start_time = datetime.datetime(2022, 5, 8, 0, 0, 0), 
                     outputfile='svd.pickle',
                     verbose=False):
    '''

#     filename = 'svd.pickle'
#     svd_analysis(record_length=1,N=24*197, # 197 = number of days from the start date until 10/5/22 when settings were changed
#                 start_time = datetime.datetime(2022, 3, 22, 0, 0, 0),
#                 f_bandstop = [-0.25,0.25])
  
# All mode 9 data / Sch+SGW:
#     filename = 'svd.pickle'
#     svd_analysis(record_length=10,
#                  dt = 10,
#                  N=24*6*65,
#                  start_time = datetime.datetime(2022, 11, 16, 0, 0, 0), # day we switched to mode 9
#                  f_bandstop = [[-0.045,0.045],[-0.5,0.5],[-0.5,0.5]],
#                  k_bandstop = [[-1, 1],[-1,-0.01],[0.01,1]] )

# All mode 9 data / SGW only
#     filename = 'svd.pickle'
#     svd_analysis(record_length=10,
#                  fs = 0.5, # Let's look at SGWs
#                  dt = 10,
#                  N=24*6*65,
#                  start_time = datetime.datetime(2022, 11, 16, 0, 0, 0), # day we switched to mode 9
#                  f_bandstop = [[-0.01,0.01],[-0.5,0.5],[-0.5,0.5]],
#                  k_bandstop = [[-1, 1],[-1,-0.02],[0.02,1]] )

# All mode 9 data / Sch only
    filename = 'svd.pickle'
    svd_analysis(record_length=10,
                 fs = 4, # Let's look at Scholte waves
                 dt = 10,
                 N=24*6*65,
                 start_time = datetime.datetime(2022, 11, 16, 0, 0, 0), # day we switched to mode 9
                 f_bandstop = [[-0.2,0.2]],
                 k_bandstop = [[-1, 1]] )
    
    '''
    Load the result and make plots
    '''
    file = open(filename, 'rb')
    U,S,V,t,f,k,nt,nx = pickle.load(file)
    file.close()
    
    for i in range(2):
        normalization = np.max(np.abs(U[:,5-i]))
        mode = np.abs(U[:,5-i].reshape((nt,nx))) / normalization
        time_series = V[5-i,:]
        variance = 100*S[5-i]/sum(S)
        plot_svd(f,k,t,mode,time_series,variance,i+1,filename=f'svd_plot_mode-{i+1}.pdf',
             flim=2,
             klim=0.04)
    
    print(f'Total runtime: {perf_counter()-t0} s')

if __name__ == "__main__":
    main()
