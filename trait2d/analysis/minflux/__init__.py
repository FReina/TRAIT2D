import pickle as pkl
import pandas as pd
import os
import numpy as np
import math as m
from scipy import optimize
import warnings
import json
from scipy import interpolate

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action = "ignore", category = SettingWithCopyWarning)
warnings.simplefilter("ignore", category = RuntimeWarning)

from trait2d.analysis import ListOfTracks
from trait2d.analysis import Track

from trait2d.analysis import ModelDB

### FUNCTIONS TO LOAD PKL FILES

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
    
###FUNCTIONS TO LOAD JSON FILES

def openJSON(path = '.', filename = ''):
    
    """
    Created on Wed July 21 14:10:17 2021
    
    @author: t.weihs
    Modified by FReina
    
    Function that reads Minflux NPY files and returns the raw output file
    -----------------
    Parameters:
        path (string, default = "."):
            the name of the folder where the json file is located
        filename (string, default = "."):
            the name of the file
    
    -----------------
    Returns:
        msr (custom): data structure containing all the localizations in the input data file
        
    """
    
    if os.path.isfile(os.path.join(path,filename)):
    
        with open(os.path.join(path,filename),'rb') as data:
            try:
                msr = json.load(data)
                data.close()
                return msr
            except:
                print('\n\tCouldn\'t read data file!\n')
                return 0
    
    else:
        print('Data not found')
        return 0

def openNPY(path = '.', filename = ''):
    """
    FReina created 08/02/22
    
    Function that reads Minflux NPY files and returns the output data as they are created by the Abberior Minflux
    -----------------
    Parameters:
        path (string, default = "."):
            the name of the folder where the npy file is located
        filename (string, default = "."):
            the name of the file
    
    -----------------
    Returns:
        msr (custom): data structure containing all the localizations in the input data file
    
    """
    
    if os.path.isfile(os.path.join(path, filename)):
        try:
            msr = np.load(os.path.join(path, filename))
            return msr
        except:
            print('Cannot read file')
            return 0
    else:
        print('File Not Found')
        return 0
    

def traceInfoFromFiles(path = '.',filename = ''):

    """
    Created on Wed July 21 14:10:17 2021
    
    @author: t.weihs
    Modified by FReina
    
    Function that reads Minflux NPY or JSON files and returns formatted data which is more easy to indicize
    -----------------
    Parameters:
        path (string, default = "."):
            the name of the folder where the npy file is located
        filename (string, default = "."):
            the name of the file
    
    -----------------
    Returns:
        traceInfo (custom): 
            data structure containing all the localizations in the input data file, with a custom data structure. 
            The "non-valid" localizations are discarded, and only the localizations emerging from the last iteration are kept.
            The data structure is organized as follows:
                - each element in traceInfo is a trace, which is a numpy. void with custom datatype;                 
                - at the first indicization (e.g., traceInfo[0]), the datatype is: [('tid', '<i4'), ('dura', '<f8'), ('nloc', '<i4'), ('mddt', '<f8'), ('mfrq', '<f8'), ('lgcy', '?'), ('trac', 'O')]
                - traceInfo[0]['trac'] is another numpy.void, with datatype [('tim', '<f8'), ('loc', '<f8', (3,)), ('fbg', '<f8'), ('frq', '<f8'), ('cts', '<i4'), ('cfr', '<f8')]
    
    """
    
    if filename.endswith("json"):
        msr = openJSON(path,filename)
    elif filename.endswith("npy"):
        msr = openNPY(path,filename)
    else:
        print("Unrecognized file format")
    
    msr = np.array(msr)
    
    traceType = [
        ('tim', float),
        ('loc', float, (3,)),
        ('fbg', float),
        ('frq', float),
        ('cts', int),
        ('cfr', float)
        ]
    
    locs = np.zeros((len(msr),), dtype = [('tid', int),
                                          ('tim', float),
                                          ('vld', bool),
                                          ('loc', float, (3,)),
                                          ('fbg', float),
                                          ('frq', float),
                                          ('cts', int),
                                          ('cfr', float)]
                    )

    for i, entry in enumerate(locs):
        locs[i]['tid'] = msr[i]['tid']
        locs[i]['tim'] = msr[i]['tim']
        locs[i]['vld'] = msr[i]['vld']
        itr = msr[i]['itr'][-1]            #itr[-1]: Only the last iteration is of interest to us (at least right now I suppose)
        locs[i]['loc'] = itr['loc']                                                  
        locs[i]['fbg'] = itr['fbg']        #background frequency during the localization, is 0.0 if bgcSense = False in the sequence
        locs[i]['frq'] = itr['efo']        #emission frequency during the localization
        locs[i]['cts'] = itr['eco']        #counts that contribute to the localization
        locs[i]['cfr'] = itr['cfr']        #center frequency ratio                               



    traceInfoType = [
        ('tid', int),
        ('dura', float),
        ('nloc', int),
        ('mddt', float),
        ('mfrq', float),
        ('lgcy', bool),
        ('trac', object),  
        ]

    LV = locs[locs['vld']]
    
    traceTidByNum, ida = np.unique(LV['tid'], return_inverse=True)
    traceNums, traceLocCntByNum = np.unique(ida, return_counts=True)

    traceInfo = np.zeros(traceNums.size, dtype=traceInfoType)
    
    try:                    #some metrics about each individual track is extracted for filtering,
                            #loc data is stored into "trac" array for each unique trace ID (tid)
    
        for n, ti in enumerate(traceInfo):
            
            ti['tid'] = traceTidByNum[n]
            idcs = np.where(LV['tid']==traceTidByNum[n])[0]
                        
            trac = np.zeros((idcs.size,), dtype = traceType)
            trac['tim'] = LV[idcs]['tim']           
            trac['loc'] = LV[idcs]['loc']
            trac['fbg'] = LV[idcs]['fbg']
            trac['frq'] = LV[idcs]['frq']
            trac['cts'] = LV[idcs]['cts']
            trac['cfr'] = LV[idcs]['cfr']
            ti['trac'] = trac
            
            trc = ti['trac']
            trl = trc[~np.isnan(trc['loc'].any(axis=1))]
            ti['dura'] = trl['tim'].max()-trl['tim'].min()          #duration of the track
            ti['nloc'] = trl.size                                   #amount of localizations in the track
            ti['mddt'] = np.nanmedian(np.diff(trl['tim']))          #median time distance in between localizations
            ti['mfrq'] = np.nanmean(trl['frq'])                     #mean time distance in between localizations
    
    except:
        print('\n\tERROR: \ttrack data wasn\'t successfully extracted!\n')
        return 0
    
    return traceInfo

def track_extractor(path = '.', filename= '',minimum_length = 0, min_frq = 0, max_frq = np.inf, factor_time_diff = 10):
    
    '''
    Extracts analyzable tracks from NPY and JSON files that come from the MINFLUX
    Can filter by frequency and trajectory length
    -----------------
    Parameters:
        path (string, default '.'):
            the name of the folder where the json file is
        filename (string):
            filename including .json extension
        minimum_length (int, default = 0):
            the minimum number of localizations in a track in order to analize it
        min_frq (int, default = 0):
            minimum value of the average emission frequency of the track for it to be considered.
            It should be left at zero unless bleaching is a concern
        max_frq (int, default = infinite):
            maximum value of the average emission frequency of the track.
            Rationale is that if the frequency ecceeds some value, it then maybe more than one emitter. 
            Can be left at infinite if that is not a concern
        factor_time_diff (int, default = 10):
            maximum time separation between two successive localizations for a track to be considered the same.
            Rationale: maybe the particle is lost by the microscope, and a different one is then tracked.
            Put it at a very high value (es: 50) to ignore
    -----------------
    Returns:
        list_of_tracks:
            list of imported trajectories, containing tracks filtered according to the input parameters            
    '''
    
    rawdata = traceInfoFromFiles(path, filename)
    #filter by minimum number of localizations. Adds +4 to avoid counting in the minimum number also the final iterations which are usually empty.
    rawdata = rawdata[rawdata['nloc']>minimum_length + 5]
    #pass these data in a pandas dataframe
    
    export = pd.DataFrame(data = [],columns = ('x','y','t','tid','frq'))
    id = 0
    
    for track in rawdata:
        id+=1
        ids = np.zeros((track['trac']['loc'][1:-4].shape[0],1))
        ids.fill(id)
        export = pd.concat([export,pd.DataFrame(np.column_stack((track['trac']['loc'][1:-4,0],track['trac']['loc'][1:-4,1],track['trac']['tim'][1:-4],ids,track['trac']['frq'][1:-4])),columns = ('x','y','t','tid','frq'))], ignore_index=True)
        
    #enact filter by time step separation and emission frequency interval
    
    list_of_tracks = []
    track_counter = 0
    
    for tid in export['tid'].unique():
        target = export.loc[export['tid']==tid]
        target['tint'] = target['t'].diff(periods = 1)
        
        #filter by time between localizations
        cuts = np.argwhere(np.array(target['tint']) > factor_time_diff * target.tint.min())
        split = np.split(target, cuts[:,0],axis = 0)
        
        #since the emission frequency is the last parameter to be checked, there is no need to use filters on the emission trace (also because it is evaluated as average)
        #if necessary, we can think of an implementation frequency-first
        for s in split:
            mean_frq = s['frq'].mean()
            if mean_frq < max_frq and mean_frq>min_frq and s.shape[0] > minimum_length:
                list_of_tracks.append({'track':s, 'tid': track_counter, 'avg_frq': mean_frq, 'length':s.shape[0]})  
                track_counter+=1
        
    return list_of_tracks

###CLASSES

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
        '''function to import using the return from the importPKL function, using legacy PKL files.
        This function does not take as input the file as minflux files are collections of tracks
        and never single.
        It can easily be turned into a hidden method in the future.
        '''    
        
        return cls(dict['track'].x,dict['track'].y,dict['track'].t,dict['tid'],dict['track'].frq)
    
    @classmethod
    def from_track_extractor(cls,dict):
        '''function to import using the return from the track_extractor function.
        This function does not take as input the file as minflux files are collections of tracks
        and never single.
        It can easily be turned into a hidden method in the future.''' 
        
        return cls(dict['track'].x,dict['track'].y,dict['track'].t,dict['tid'],dict['track'].frq)
        
    
    def plot_msd(self):
        t = self._tn
        msd = self._msd
        err = self._msd_error
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid(linestyle='dashed', color='grey')
        plt.xlabel("t")
        plt.ylabel("MSD")
        plt.semilogx(t, msd, color='black')
        plt.fill_between(t, msd-err, msd+err, color='black', alpha=0.5)
        
    def plot_adc(self):
        t = self._tn
        adc = self._adc
        err = self._adc_error
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid(linestyle='dashed', color='grey')
        plt.xlabel("t")
        plt.ylabel("ADC")
        plt.semilogx(t, adc, color='black')
        plt.fill_between(t, adc-err, adc+err, color='black', alpha=0.5)
    
    def MF_calculate_adc(self, R: float = 1/6, precision = 5):
        '''calculate Apparent Diffusion Coefficient with binning, following a less sophisticated method arising from discussion with TW.
        
        precision: number of decimal places to consider when binning the time intervals, higher precision may lead to slower
        computation times.'''
    
        if precision <4:
            raise Exception("not enough decimal spaces")
        else:
            if precision >8:
                raise warnings.warn("this level of precision does not make sense")
        
        if self._msd is None:
            self.MF_calculate_msd()
            
        dt = self._tn[0]
        N = self._msd.size
    
        #time array
    
        T = self._tn[0:N+1]    
            
        self._adc = self._msd / (4 * T * (1-2*R*dt / T))
        self._adc_error = self._msd_error / (4 * T * (1 - 2*R*dt / T))
        
    def MF_calculate_msd_nonmatrix(self, precision = 5):
        '''calculate MSD with binning, following a less sophisticated method arising from discussion with TW.
        
        precision: number of decimal places to consider when binning the time intervals, higher precision may lead to slower
        computation times.'''
        
        if precision <4:
            raise Exception("not enough decimal spaces")
        else:
            if precision >8:
                raise warnings.warn("this level of precision does not make sense")
        
        #calculate arrays with all possible time intervals and all possible squared displacements.
        #in combinatorics, these are all the combinations of values of the time array, by two elements
        from itertools import combinations
        from math import factorial
        
        #zip x and y to obtain an ndarray of localization (x and y as tuples)
        locs = np.array(list(zip(self._x,self._y)))
        #create pairs of times and localizations. The combinations function in itertools always pairs in the same way, so this passage preserves the order
        time_pairs = np.array(list(combinations(self._t, 2)))
        x_pairs = np.array(list(combinations(self._x,2)))
        y_pairs = np.array(list(combinations(self._y,2)))
        
        #create arrays of time lags and squared displacements. Use preallocation for speed
        #time_intervals = np.zeros(factorial(len(self._t))/factorial(2)/factorial(len(self._t)-2))
        #sq_disp = np.zeros(factorial(len(self._t))/factorial(2)/factorial(len(self._t)-2))  
        
        time_intervals = np.array(list(map(lambda t:t[1]-t[0], time_pairs)))
        x_disp = np.array(list(map(lambda x:(x[1]-x[0])**2, x_pairs)))
        y_disp = np.array(list(map(lambda y:(y[1]-y[0])**2, y_pairs)))    
        #sq_disp = np.array(list(map(lambda p:((p[2]-p[0])**2+(p[3]-p[1])**2),locs_pairs)))
        #sq_disp = np.array([(p[1][0]-p[0][0])**2+(p[1][1]-p[0][1])**2 for p in locs_pairs])
        sq_disp = np.array([xd+yd for xd,yd in zip(x_disp,y_disp)])
        
        #sort time lags and displacements according to the former
        
        #time_intervals = np.array([t for t in sorted(time_intervals)])
        #sq_disp = np.array([d for t,d in sorted(zip(time_intervals,sq_disp), key = lambda loc: loc[0])])
        
        #no need to look for minimum value of time intervals. The number of bins can be found easily
        
        min_timeint = np.amin(time_intervals)
        
        nbins = m.floor((self._t[-1]-self._t[0])/min_timeint)
        #and the bin centers. We divide the time dimension in equally spaced bins, but they are later readjusted
        bin_centers = np.linspace(min_timeint, min_timeint*nbins,num = nbins)
        
        #the next step should be much faster than previous implementations.
        #The index is much more efficient when the number of displacements corresponding to a specific time interval is known. This is not the case for MINFLUX necessarily

        self._msd = np.empty(0)
        self._tn = np.empty(0)
        self._msd_error= np.empty(0)
        self._tn_error = np.empty(0) 
        nn = np.empty(0)       
        
        for center in bin_centers:
            #now for every bin_center we need to make a mask on time_intervals.
            #They are binned so that the left is included and the right is excluded
            idx = (time_intervals>=center-min_timeint/2) & (time_intervals<center+min_timeint/2)
            nn = np.append(nn, len(np.nonzero(idx)[0]))
            #check if a bin is not empty
            if not nn[-1]==0:    
                self._msd = np.append(self._msd, np.mean(sq_disp[idx]))
                self._tn = np.append(self._tn,np.mean(time_intervals[idx]))
                self._msd_error = np.append(self._msd_error, np.std(sq_disp[idx])/nn[-1])
                self._tn_error = np.append(self._tn_error,np.std(time_intervals[idx])/nn[-1])
                     
    

    def MF_calculate_msd(self, precision=5):
        '''calculate MSD with binning, following a less sophisticated method arising from discussion with TW.
        
        precision: number of decimal places to consider when binning the time intervals, higher precision may lead to slower
        computation times.'''
        
        if precision <4:
            raise Exception("not enough decimal spaces")
        else:
            if precision >8:
                raise warnings.warn("this level of precision does not make sense")
                
        #calculate matrices to get numbers from
        #time interval matrices
        tint_matrix = self._t.reshape(-1,1) - self._t #calculate all possible positive time intervals as a matrix
        time_intervals = np.tril(tint_matrix) #we only really need the lower triangular matrix
        
        #we now need to create the corresponding squared displacements
        xdis_matrix = self._x.reshape(-1, 1) - self._x
        ydis_matrix = self._y.reshape(-1, 1) - self._y
        sdis_matrix = np.power(xdis_matrix, 2.0) + np.power(ydis_matrix, 2.0) # squared displacement matrix
        displacements = np.tril(sdis_matrix) # only using the lower triangular matrix coherently with the time_intervals matrix above
        
        #the binning is done on the linear array of unique time intervals 
        
        unique_time_intervals, cardinality = np.unique(time_intervals[np.nonzero(time_intervals)], return_counts=True)
        #we get the minimum time interval
        min_timeint = np.amin(unique_time_intervals[np.nonzero(unique_time_intervals)])
        #get the number of bins as an approximation with the total duration of the track divided by the min_timeint
        nbins = m.floor((self._t[-1]-self._t[0])/min_timeint)
        #now we have to calculate the bin centers
        bin_centers = np.linspace(min_timeint, min_timeint*nbins,num = nbins)
        
        #now for every bin_center we need to make a mask on time_intervals
        
        self._msd = np.empty(0)
        self._tn = np.empty(0)
        self._msd_error= np.empty(0)
        self._tn_error = np.empty(0)
        nn = np.empty(0)
        
        for center in bin_centers:
            #now for every bin_center we need to make a mask on time_intervals.
            #They are binned so that the left is included and the right is excluded
            idx = (time_intervals>=center-min_timeint/2) & (time_intervals<center+min_timeint/2)
            #test the correctness of the binning
            nn = np.append(nn, len(np.nonzero(idx)[0]))
            #check if a bin is not empty
            if not nn[-1]==0:            
                #the rest is easy
                self._msd = np.append(self._msd,np.mean(sdis_matrix[idx]))
                self._msd_error = np.append(self._msd_error,np.std(sdis_matrix[idx])/nn[-1])
                self._tn = np.append(self._tn,np.mean(tint_matrix[idx]))
                self._tn_error = np.append(self._tn_error,np.std(tint_matrix[idx])/nn[-1])
        
        
    from trait2d.analysis.minflux._msd import MF_msd_analysis
    from trait2d.analysis.minflux._adc import MF_adc_analysis
            
    def _MF_categorize(self, Dapp, J, Dapp_err = None, R: float = 1/6, fraction_fit_points: float = 0.25, fit_max_time: float=None, maxfev=1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
        #J is an array (e.g. numpy.arange(0,N), with N being the largest index that will be used for the analysis) that selects the points that will be fit with the models
        
        if fraction_fit_points > 0.25:
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")
            
        #define time array and time interval a bit differently

        T = self._tn[J]
        dt = self._tn[0]
        

        # Get number of points for fit from either fit_max_time or fraction_fit_points
        if fit_max_time is not None:
            n_points = int(np.argwhere(T < fit_max_time)[-1])
        else:
            n_points = np.argmax(J > fraction_fit_points * J[-1])  
        
        cur_dist = 0
        idxs = []
            
        if enable_log_sampling:
            # Get indexes that are (approximately) logarithmically spaced
            idxs.append(0)
            for i in range(1, n_points):
                cur_dist += np.log10(T[i]/T[i-1])
                if cur_dist >= log_sampling_dist:
                    idxs.append(i)
                    cur_dist = 0
        else:
            # Get every index up to n_points
            idxs = np.arange(0, n_points, dtype=int)
        
        error = None
        if not Dapp_err is None:
            error = Dapp_err[idxs]        
        # Perform fits for all included models
        fit_results = {}

        bic_min = 999.9
        category = None
        sigma = None
        if weighting == 'error':
            sigma = error
        elif weighting == 'inverse_variance':
            sigma = np.power(error, 2.0)
        elif weighting == 'variance':
            sigma = 1 / np.power(error, 2.0)
        elif weighting == 'disabled':
            sigma = None
        else:
            raise ValueError("Unknown weighting method: {}. Possible values are: 'error', 'variance', 'inverse_variance', and 'disabled'.".format(weighting))

        for model in ModelDB().models:
            model.R = R
            model.dt = dt
            model_name = model.__class__.__name__

            r = optimize.curve_fit(model, T[idxs], Dapp[idxs], p0 = model.initial,
                        sigma = sigma, maxfev = maxfev, method='trf', bounds=(model.lower, model.upper), absolute_sigma = True)
            perr = np.sqrt(np.diag(r[1]))
            pred = model(T, *r[0])
            bic = BIC(pred[idxs], Dapp[idxs], len(r[0]), len(idxs))
            if bic < bic_min:
                bic_min = bic
                category = model_name

            from scipy.stats import kstest
            test_results = kstest(Dapp[idxs], pred[idxs], N = len(idxs))

            fit_results[model_name] = {"params": r[0], "errors": perr, "bic" : bic, "KSTestStat": test_results[0], "KStestPValue": test_results[1]}

        # Calculate the relative likelihood for each model
        for model in ModelDB().models:
            model_name = model.__class__.__name__
            rel_likelihood = np.exp((-fit_results[model_name]["bic"] + bic_min) * 0.5)
            fit_results[model_name]["rel_likelihood"] = rel_likelihood

        fit_indices = idxs
        return category, fit_indices, fit_results
    
    
class MFResTrack(MFTrack):
    ''''
    A class to hold only tn, MSD, Dapp and relative error. Analysis routines will work, but other that require the coordinates of the particles will obviously not work
    '''
    
    def __init__(self, tn, tn_err, msd, msd_err, Dapp, Dapp_err):
        self._tn = tn
        self._tn_error= tn_err
        self._msd = msd
        self._msd_error = msd_err
        self._adc = Dapp
        self._adc_error = Dapp_err
        
    def MF_calculate_adc(self):
        TypeError('Cannot use this method on a MFResTrack')

    def MF_calculate_msd(self):
        TypeError('Cannot use this method on a MFResTrack')
        
    def calculate_msd(self):
        TypeError('Cannot use this method on a MFResTrack')
        
    def plot_trajectory(self):
        TypeError('Cannot use this method on a MFResTrack')
        
    def __repr__(self):
        return ("<%s instance at %s>\n"
                "------------------------\n"
                "Track length:%s\n"
                "Track ID:%s\n"
                "------------------------\n"
                "MSD analysis done:%s\n"
                "ADC analysis done:%s\n") % (
            self.__class__.__name__,
            id(self),
            str(self._t.size).rjust(11, ' '),
            str(self._id).rjust(15, ' '),
            str(self._msd_analysis_results is not None).rjust(6, ' '),
            str(self._adc_analysis_results is not None).rjust(6, ' ')
        )
    

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
    
    @classmethod
    def from_track_extractor(cls, path, name, minimum_length = 100, min_frq = 0, max_frq = np.inf, factor_time_diff = 10):
        """Import all trajectories from a Minflux .npy file into a MFTrackDB object.
        Usage: MFTrackDB.from_track_extractor(**kwargs).
        Can filter and split trajectories by photon emission frequency, trajectory length, and time lag between localizations.
    -----------------
    Parameters:
        path (string, default '.'):
            the name of the folder where the json file is
        filename (string):
            filename including .json extension
        minimum_length (int, default = 0):
            the minimum number of localizations in a track in order to analize it
        min_frq (int, default = 0):
            minimum value of the average emission frequency of the track for it to be considered.
            It should be left at zero unless bleaching is a concern
        max_frq (int, default = infinite):
            maximum value of the average emission frequency of the track.
            Rationale is that if the frequency ecceeds some value, it then maybe more than one emitter. 
            Can be left at infinite if that is not a concern
        factor_time_diff (int, default = 10):
            maximum time separation between two successive localizations for a track to be considered the same.
            Rationale: maybe the particle is lost by the microscope, and a different one is then tracked.
            Put it at a very high value (es: 50) to ignore
    -----------------
    Returns:
        list_of_tracks:
            list of imported trajectories, containing tracks filtered according to the input parameters
        """
    
        ensemble = []
        if name.endswith('npy') or name.endswith('json'):
            rawdata = track_extractor(path=path,filename = name,minimum_length = minimum_length, min_frq = min_frq, max_frq = max_frq, factor_time_diff = factor_time_diff)
        
        for dataset in rawdata:
            ensemble.append(MFTrack.from_track_extractor(dataset))
            
        return cls(ensemble)
        
    def MF_calculate_msd(self):
        """Calculate the MSD for all the MFTrack objects present in the MFTrackDB object."""
        
        for track in self._tracks:
            track.MF_calculate_msd()
        
        return None
    
    def msd_analysis(self, **kwargs):
        """Analyze all tracks using MSD analysis.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be used by msd_analysis for each track.

        Returns
        -------
        list_failed
            List containing the indices of the tracks for which the analysis failed.
        """
        list_failed = []
        i = 0
        for track in self._tracks:
            try:
                track.MF_msd_analysis(**kwargs)
            except:
                list_failed.append(i)
            i += 1

        if len(list_failed) > 0:
            warnings.warn("MSD analysis failed for {}/{} tracks. \
                Consider raising the maximum function evaluations using \
                the maxfev keyword argument. \
                To get a more detailed stacktrace, run the MSD analysis \
                for a single track.".format(len(list_failed), len(self._tracks)))

        return list_failed
    
    
    def adc_analysis(self,**kwargs):
        """Analyze all tracks using ADC analysis.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be used by MF_adc_analysis for each track.

        Returns
        -------
        list_failed
            List containing the indices of the tracks for which the analysis failed.
        """
        list_failed = []
        i = 0
        for track in self._tracks:
            try:
                track.MF_adc_analysis(**kwargs)
            except:
                list_failed.append(i)
            i+=1
        
        if len(list_failed) > 0:
            warnings.warn("ADC analysis failed for {}/{} tracks. "
                "Consider raising the maximum function evaluations using "
                "the maxfev keyword argument. "
                "To get a more detailed stacktrace, run the ADC analysis "
                "for a single track.".format(len(list_failed), len(self._tracks)))

        return list_failed
    
    def MF_ensemble_average(self, max_index = -1, perform_analysis = False, **kwargs):
        '''' 
        Calculate the ensemble average of MSD and ADC for all the tracks in the MINFLUX framework (meaning: the time tn also gets averaged).
        The ADC gets analyzed through the ADC analysis pipeline as well.
        This module works in the assumption that the values for tn do not change too much across tracks.
        Note: this calculates also the ensemble average MSD and relative error, but DOES NOT run the MSD analysis
        
        Parameters
        ----------------
        max_index (int, default = -1)
            In case the user wants a specific number of MSD and ADC points
            
        perform_analysis: (bool, default: False)
            Flag to control whether to run MF_adc_analysis on the ensemble average.
            
        **kwargs: keyword arguments to be used by MF_adc_analysis
        
        Returns
        ----------------
        
        EAverage : MFResTrack
            MFResTrack object containing the ensemble average of all tracks. The type of object is due to the lack of coordinates in this case
        
        results: dict
            If perform_analysis is set to true, exports this dictionary as well.
            Contains the variable ensemble average for all tracks and the results of the ADC analysis for the ensemble average ADC.
            Error defined as the standard error of the mean (ddof = 0)
            
        '''
        #check if the MSD and ADC analysis are calculated and raise exception if it is not done for some of the tracks
        
        for track in self._tracks:
            if track._msd is None and track._adc is None:
                raise ValueError('the MSD or ADC for one of the tracks in the database is not calculated.')
        
          
        import pandas as pd
        
        #we will make use of the pandas dataframe modules as they provide a fast way to skip empty values
        #it is important to make sure that the tn are similar across all tracks
        
        #make pandas dataframes of tn, MSD and ADC
        
        tn_df = pd.DataFrame(data = [track._tn[0:max_index] for track in self._tracks]).transpose()
        msd_df = pd.DataFrame(data = [track._msd[0:max_index] for track in self._tracks]).transpose()
        adc_df = pd.DataFrame(data = [track._adc[0:max_index] for track in self._tracks]).transpose()
        
        tn_ea = tn_df.mean(axis=1,skipna=True).to_numpy()
        tn_err_ea = tn_df.sem(ddof=0,axis=1,skipna=True).to_numpy()
        msd_ea = msd_df.mean(axis=1,skipna=True).to_numpy()
        msd_err_ea = msd_df.sem(axis=1,ddof=0,skipna=True).to_numpy()
        adc_ea = adc_df.mean(axis=1,skipna=True).to_numpy()
        adc_err_ea = adc_df.sem(axis=1,ddof=0,skipna=True).to_numpy()
                
        
        EAverage = MFResTrack(tn = tn_ea,tn_err = tn_err_ea,msd = msd_ea, msd_err = msd_err_ea,Dapp = adc_ea, Dapp_err = adc_err_ea)
        
        if perform_analysis:
            ea_adc_results = {}
        
            ea_adc_results = EAverage.MF_adc_analysis(**kwargs)
        
            del ea_adc_results["Dapp"]
            del ea_adc_results["Dapp_err"]      
                
            return {"average_tn": tn_df.mean(axis=1,skipna=True).tolist(),
                    "average_tn_err": tn_df.sem(axis = 1, ddof=0,skipna=True).tolist(),
                    "average_msd": msd_df.mean(axis=1,skipna=True).tolist(),
                    "average_msd_err": msd_df.sem(axis=1,ddof=0,skipna=True).tolist(),
                    "average_dapp": adc_df.mean(axis=1,skipna=True).tolist(),
                    "average_dapp_err": adc_df.sem(axis=1,ddof=0,skipna=True).tolist(),
                    "adc_results": ea_adc_results}, EAverage
        else:
            return EAverage
    
    
    def MF_model_average(self, max_index = -1, perform_analysis = False, **kwargs):
        '''' 
        Calculate the ensemble average of MSD and ADC for all the tracks in the MINFLUX framework (meaning: the time tn also gets averaged).
        The ADC gets analyzed through the ADC analysis pipeline as well.
        The routine skips over unidentified tracks
        
        Parameters
        ----------------nm
        max_index (default = -1)
            In case the user wants a specific number of MSD and ADC points
            
        perform_analysis: (bool, default: False)
            Flag to control whether to run MF_adc_analysis on the ensemble average of the tracks divided by model. 
            If true, it will run MF_adc_analysis on the average of the tracks segmented by the same model.
            
        **kwargs
            Arguments of MF_adc_analysis
        
        Returns
        ----------------
        results: dict
            Contains the variable averages by model, the results of the ADC analysis for the average ADCs (all models), with the same output style as MF_ensemble_average.
            The keys are the model names. 
            Each key also contains a dump of the trajectories as a MFTrackDB.
        '''        
        #check if the ADC analysis has been carried out on the tracks
        
        for track in self._tracks:
            if track._adc_analysis_results is None:
                raise ValueError('the ADC analysis for all tracks needs to be carried out first')
        
        results = {}
        
        for model in ModelDB().models:
            modname= model.__class__.__name__
            #get a list and then a MFTrackDB of the tracks which are best described by model
            model_ensemble = [track for track in self._tracks if track._adc_analysis_results["best_model"] == modname]  
            if model_ensemble:
                results[modname] = {}
                segmented = MFTrackDB(model_ensemble)
                if perform_analysis: #do we want to run the analysis again?
                    if len(model_ensemble) == 1:
                        results[modname] = segmented._tracks[0].MF_adc_analysis(**kwargs)
                    else: 
                        results[modname] = segmented.MF_ensemble_average(**kwargs)
                trackdump = {'tracks' : segmented}
                results[modname].update(trackdump)     
            
            
        pie = {'sectors': [(modname, len(results[modname]['tracks']._tracks)) for modname in results.keys()]}
        
        results.update(pie)
        
        return results
    
    def adc_summary(self,**kwargs):
        
        print('Use MF_adc_summary instead!')
        
    def MF_adc_summary(self, max_index = -1, avg_only_params = False, ensemble_average = False, plot_msd = False, plot_dapp = False, plot_pie_chart = False):
        """Average tracks by model and optionally plot the results.
        This is the Minflux version. That means that the tracks potentially have very different lengths. 
        It makes use of the MF_model_average module heavily.

        Parameters
        ----------
        max_index (default = -1)
            In case the user wants a specific number of MSD and ADC points
        avg_only_params: bool
            Only average the model parameters but not D_app and MSD. If this flag is True, the function will only return the fit parameters as a dictionary, divided by best model, the means and the standard deviation
        ensemble_average: (bool, default = False)
            runs MF_ensemble_average(max_index = max_index, perform_analysis=False) and adds the ensemble average of tn, msd and adc of the tracks to the return of the function
        plot_msd: bool
            Plot the averaged MSD for each model, and the ensemble average.
        plot_dapp: bool
            Plot the averaged D_app for each model, and the ensemble average.
        plot_pie_chart: bool
            Plot a pie chart showing the relative fractions of each model.

        Returns
        -------
        results : dict
            Without other options, it will return MF_model_average()
        """
        warnings.warn('MODEL UNDER CONSTRUCTION, PLOTS STILL NOT DONE')
        
        #Check if the ADC analysis has been run on all tracks
        
        for track in self._tracks:
            if track._adc_analysis_results is None:
                raise ValueError('the ADC analysis for all tracks needs to be carried out first')
        
        #set up parameter exporting
        
        results = {}
                
        if avg_only_params:
            
            params = {}
            params['sectors'] = self.MF_model_average()['sectors']
            for model in ModelDB().models:
                modname = model.__class__.__name__
                #get a list and then a MFTrackDB of the tracks which are best described by model
                model_ensemble = [track for track in self._tracks if track._adc_analysis_results["best_model"] == modname]
                params[modname] = {}
                if model_ensemble:
                    segmented = MFTrackDB(model_ensemble)
                    params[modname]['data'] = np.stack([track._adc_analysis_results['fit_results'][modname]['params'] for track in segmented._tracks], axis = 0)
                    params[modname]['average'] = [np.average(params[modname]['data'][:,i]) for i in range(params[modname]['data'].shape[1])]
                    params[modname]['stdev'] = [np.std(params[modname]['data'][:,i]) for i in range(params[modname]['data'].shape[1])]
            
            results = params
                    
        else:
            results = self.MF_model_average()
                
        if ensemble_average:
            results = self.MF_ensemble_average(max_index = max_index, perform_analysis=False)
            results.update(self.MF_model_average(max_index = max_index, perform_analysis=False))
        else:
            results = self.MF_model_average()
        
        return results
    
    def pop_track(self, track_id = -1):
        """Method that eliminates a track from the total, and updates the MFTrackDB. The indeces will also update to avoid gaps.
        
        Parameters
        ----------
        track_id (default = -1)
            selects which track to eliminate by the index
            
        """
        warnings.warn('Please update the ensemble and segmented averages!')
                
        self._tracks.pop(track_id)
        for counter, track in enumerate(self._tracks):
            track._id = counter
        return MFTrackDB(self._tracks)
        
    


def BIC(pred: list, target: list, k: int, n: int):
    """Bayesian Information Criterion
    Parameters
    ----------
    pred: list
        Model prediction
    target: list
        Model targe

    k: int
        Number of free parameters in the fit
    n: int
        Number of data points used to fit the model
    Returns
    -------
    bic : float
        Bayesian Information Criterion
    """
    # Compute RSS
    RSS = np.sum((np.array(pred) - np.array(target)) ** 2)
    bic = k * np.log(n) + n * np.log(RSS / n)
    return bic
    