import pickle as pkl
import pandas as pd
import os
import numpy as np

from trait2d.analysis import ListOfTracks
from trait2d.analysis import Track

def openPKL(path = '.',name=''): #name must be the name of the *pkl-file, e. g. 'Bilayer2_low-conc_meas10_L100.msr.pkl'
    '''Legacy opener for old Minflux PKL files'''
    with open(os.path.join(path,name),'rb') as f:
        traceInfo = pkl.load(f)
    return traceInfo

def importPKL(path='.',name='',minimum_length = 500, min_frq = 0, max_frq = 250000, factor_time_diff = 5):
        '''Legacy importer for old Minflux PKL files.
        Filters by minimum_length and min_frq and max_frq'''
        #use openPKL to get the data from the file
        rawdata = openPKL(path,name)
        if type(rawdata) == list:
            rawdata = rawdata[0]
        #filter by minimum number of localizations. Add a +4 to avoid counting in the minimum number also the final numbers that are usually empty. 
        data_filtered = rawdata[rawdata['nloc'] > minimum_length+5]
        #initialize dataframe for exporting
        export = pd.DataFrame(data = [], columns = ('x','y','t','tid','frq'))
        
            #put filtered data in the pandas dataframe
        for i in range(0,len(data_filtered['tid'])):
            ids = np.zeros((data_filtered['trac'][i]['loc'][1:-4].shape[0],1))
            ids.fill(i)
            #cuts the first localization because it is always at a longer time interval from the second.
            export = export.append(pd.DataFrame(np.column_stack((data_filtered['trac'][i]['loc'][1:-4],data_filtered['trac'][i]['tim'][1:-4],ids,data_filtered['trac'][i]['frq'][1:-4])),columns = ('x','y','t','tid','frq')))
        
        #filter by maximum time separation between two successive localizations, interval of frequency
        
        list_of_tracks = []
        track_counter = 0
        
        for tid in export['tid'].unique():
            target = export.loc[export['tid']==tid]
            target['tint'] = target['t'].diff(periods = 1)
            
            #filter by time between localizations
            cuts = np.argwhere(np.array(target['tint']) > factor_time_diff * target.tint.min())
            split = np.split(target, cuts[:,0],axis = 0)
            
            for s in split:
                mean_frq = s['frq'].mean()
                if mean_frq < max_frq and mean_frq>min_frq and s.shape[0] > minimum_length:
                    list_of_tracks.append({'track':s, 'tid': track_counter, 'avg_frq': mean_frq, 'length':s.shape[0]})
                    track_counter+=1
                    
        return list_of_tracks

class MFTrack(Track):
    '''custom class to work with Minflux tracks in the legacy PKL format.
    For now, it works exclusively with the output format from importPKL'''
    def __init__(self,x=None,y=None,t=None,id=None,frq=None):
        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._t = np.array(t, dtype=float)

        self._id = id
        self._frq = frq

        self._tn = None #since we are going to bin the time, we need an additional time array
        self._tn_error = None
        
        self._msd = None
        self._msd_error = None
        self._adc = None
        self._adc_error = None

        self._msd_analysis_results = None
        self._adc_analysis_results = None  
        
    @classmethod
    def from_importPKL(cls,dict):
        '''function to import using the return from the importPKL function, using legacy PKL files'''    
        
        return cls(dict['track'].x,dict['track'].y,dict['track'].t,dict['tid'],dict['track'].frq)
    
    def MF_calculate_msd(self, use_log = False,mod_factor = 0.3):
        '''calculate MSD according to the method elaborated by FR, 
        which includes KMeans clustering to bin the squared displacements 
        given the uneven nature of the sampling
        The computation time can be improved with use_log = True UNTESTED
        mod_factor is set to 0.3 as default. Increasing it will increase the number of clusters (i.e. time points), 
        but it starts behaving strangely, increasing variability in msd, and disregarding short time intervals. 
        With anything more than 0.8 (actually not rigorously tested), set use_log = True
        '''
        from sklearn.cluster import KMeans
        import time
        
        tint_matrix_og = self._t.reshape(-1,1) - self._t #calculate all possible positive time intervals as a matrix
        tint_matrix = np.around(self._t.reshape(-1,1) - self._t,5) #rounded down version to speed up computations
        time_intervals = np.tril(tint_matrix) #we only really need the lower triangular matrix
        #the rounding operation to make sure we are not too sensitive to time differences
        
        #we now need to create the corresponding squared displacements
        xdis_matrix = self._x.reshape(-1, 1) - self._x
        ydis_matrix = self._y.reshape(-1, 1) - self._y
        sdis_matrix = np.power(xdis_matrix, 2.0) + np.power(ydis_matrix, 2.0) # squared displacement matrix
        displacements = np.tril(sdis_matrix) #
        
        #the displacements matrix and the time intervals matrix have the same shape and they are both triangular. This way we can indicize them easily to obtain MSD
        
        #we will only cluster the unique values of the time intervals for speed. The counts are used to weigh the clustering
        
        unique_time_intervals, unique_counts = np.unique(time_intervals[np.nonzero(time_intervals)], return_counts=True)
        
        #CLUSTERING STEP
        #The computation time can be improved by using the log-version
        
        #preliminary parameters
        original_shape = tint_matrix.shape #necessary for the indexing operation
        min_timeint = np.amin(unique_time_intervals[np.nonzero(unique_time_intervals)])
        #initialize initial guess for the clustering and number of clusters
        initial_guess = np.linspace(min_timeint, min_timeint*int(mod_factor*len(self._t)),num = int(mod_factor*len(self._t)))
        n_clusters = len(initial_guess) #avoids segfaults
        
        if not use_log:
            classification = KMeans(n_clusters = n_clusters, init = initial_guess.reshape(-1,1), max_iter= 2000, algorithm='full',n_init = 1).fit(unique_time_intervals.reshape(-1,1),sample_weight=unique_counts)
        else:
            #THIS NEEDS TO BE TESTED
            classification = KMeans(n_clusters = n_clusters, init = np.log(initial_guess.reshape(-1,1)), max_iter= 2000, algorithm='full',n_init = 1).fit(np.log(unique_time_intervals.reshape(-1,1)),sample_weight=unique_counts)
            #classification.cluster_centers_ = np.exp(classification.cluster_centers_) #bring everything to regular scale again
            
        #actual calculation of a proper time array (maybe it can be cut to save time), msd and msd_error. This is the most computationally heavy step
        self._msd=[]
        self._tn = []
        self._msd_error= []
        self._tn_error = []
        from tqdm import tqdm
        
        for i in tqdm(range(max(classification.labels_))):
            cluster_idx = np.where(classification.labels_==i) #select which of the unique_time_intervals are to be considered now
            #now we select the indices of the time_intervals matrix (lower triangular) fall into the cluster selected above
            idx = np.where((time_intervals>=np.min(unique_time_intervals[cluster_idx]))&(time_intervals<=np.max(unique_time_intervals[cluster_idx])))
            #the calculation of the MSD is now trivial 
            self._msd.append(np.mean(sdis_matrix[idx]))
            self._msd_error.append(np.std(sdis_matrix[idx]))
            self._tn.append(np.mean(tint_matrix_og[idx]))
            self._tn_error.append(np.std(tint_matrix_og[idx]))
        

class MFTrackDB(ListOfTracks):
    '''A custom class to work with the Minflux Tracks in the legacy PKL format'''
    
    def __init__(self, tracks: list):
        self._tracks = tracks
        
    def __repr__(self):
        return ("<%s instance at %s>\n"
                "Number of tracks: %s\n") % (self.__class__.__name__,id(self),len(self._tracks))   
        
    def get_track_id(self,idx):
        for i in range(len(self._tracks)):
            if self._tracks[i]['tid'] == idx:
                return self._tracks[i]
            
    @classmethod
    def from_pkl(cls, path, name, minimum_length = 500, min_frq = 0, max_frq = 250000, factor_time_diff = 5):
        
        ensemble = []
        rawdata = importPKL(path=path,name = name,minimum_length = minimum_length, min_frq = min_frq, max_frq = max_frq, factor_time_diff = factor_time_diff)
        
        for dataset in rawdata:
            ensemble.append(MFTrack.from_importPKL(dataset))
            
        return cls(ensemble)
    