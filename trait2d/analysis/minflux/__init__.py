import pickle as pkl
import pandas as pd
import os
import numpy as np
import math as m
from scipy import optimize
import numpy as np
import warnings
import tqdm
from scipy import interpolate

from trait2d.analysis import ListOfTracks
from trait2d.analysis import Track

from trait2d.analysis import ModelDB

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
    
    def MF_calculate_msd_cluster(self, use_log = False,mod_factor = 0.3):
        '''calculate MSD according to the method elaborated by FR, 
        which includes KMeans clustering to bin the squared displacements 
        given the uneven nature of the sampling
        The computation time can be improved with use_log = True 
        mod_factor is set to 0.3 as default. Increasing it will increase the number of clusters (i.e. time points), 
        but it starts behaving strangely, increasing variability in msd, and disregarding short time intervals. 
        With anything more than 0.8 (actually not rigorously tested), set use_log = True
        '''
        from sklearn.cluster import KMeans
        
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
        self._msd = np.empty(0)
        self._tn = np.empty(0)
        self._msd_error= np.empty(0)
        self._tn_error = np.empty(0)
        cardinality = np.empty(0)
        
        from tqdm import tqdm
        
        for i in tqdm(range(max(classification.labels_))):
            cluster_idx = (classification.labels_==i) #select which of the unique_time_intervals are to be considered now
            #now we select the indices of the time_intervals matrix (lower triangular) fall into the cluster selected above
            idx = np.where((time_intervals>=np.min(unique_time_intervals[cluster_idx]))&(time_intervals<=np.max(unique_time_intervals[cluster_idx])))
            #the calculation of the MSD is now trivial
                
            cardinality = np.append(cardinality,len(idx[0]))
            self._msd = np.append(self._msd,np.mean(sdis_matrix[idx]))
            self._msd_error = np.append(self._msd_error,np.std(sdis_matrix[idx])/np.sqrt(len(idx[0])))
            self._tn = np.append(self._tn,np.mean(tint_matrix_og[idx]))
            self._tn_error = np.append(self._tn_error,np.std(tint_matrix_og[idx])/np.sqrt(len(idx[0])))
        
        return initial_guess, classification, cardinality
    
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
                        sigma = sigma, maxfev = maxfev, method='trf', bounds=(model.lower, model.upper))
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
    
    def MF_ensemble_average(self, max_index = -1, **kwargs):
        '''' 
        Calculate the ensemble average of MSD and ADC for all the tracks in the MINFLUX framework (meaning: the time tn also gets averaged).
        The ADC gets analyzed through the ADC analysis pipeline as well.
        This module works in the assumption that the values for tn do not change too much across tracks.
        Note: this calculates also the ensemble average MSD and relative error, but DOES NOT run the MSD analysis
        
        Parameters
        ----------------
        max_index (default = -1)
            In case the user wants a specific number of MSD and ADC points
            
        **kwargs: keyword arguments to be used by MF_adc_analysis
        
        Returns
        ----------------
        results: dict
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
        msd_err_ea = msd_df.sem(axis=1,skipna=True).to_numpy()
        adc_ea = adc_df.mean(axis=1,skipna=True).to_numpy()
        adc_err_ea = adc_df.sem(axis=1,skipna=True).to_numpy()
                
        
        EAverage = MFResTrack(tn = tn_ea,tn_err = tn_err_ea,msd = msd_ea, msd_err = msd_err_ea,Dapp = adc_ea, Dapp_err = adc_err_ea)
        
        ea_adc_results = EAverage.MF_adc_analysis(**kwargs)
        
        del ea_adc_results["Dapp"]
        del ea_adc_results["Dapp_err"]      
        
        
        return {"average_tn": tn_df.mean(axis=1,skipna=True).tolist(),
                "average_tn_err": tn_df.sem(axis = 1, ddof=0,skipna=True).tolist(),
                "average_msd": msd_df.mean(axis=1,skipna=True).tolist(),
                "average_msd_err": msd_df.sem(axis=1,ddof=0,skipna=True).tolist(),
                "average_dapp": adc_df.mean(axis=1,skipna=True).tolist(),
                "average_dapp_err": adc_df.sem(axis=1,ddof=0,skipna=True).tolist(),
                "adc_results": ea_adc_results}
    
    
    def MF_model_average(self, max_index = -1, **kwargs):
        '''' 
        Calculate the ensemble average of MSD and ADC for all the tracks in the MINFLUX framework (meaning: the time tn also gets averaged).
        The ADC gets analyzed through the ADC analysis pipeline as well.
        The routine skips over unidentified tracks
        
        Parameters
        ----------------nm
        max_index (default = -1)
            In case the user wants a specific number of MSD and ADC points
            
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
                if len(model_ensemble) == 1:
                    results[modname] = segmented._tracks[0].adc_analysis(**kwargs)
                else: 
                    results[modname] = segmented.MF_ensemble_average(**kwargs)
                trackdump = {'tracks' : segmented}
                results[modname].update(trackdump)     
            
            
        pie = {'sectors': [(modname, len(results[modname]['tracks']._tracks)) for modname in results.keys()]}
        
        results.update(pie)
        
        return results
    
    def adc_summary(self,**kwargs):
        
        print('Use MF_adc_summary instead!')
    
    def MF_adc_summary(self, avg_only_params = False, interpolation = False, plot_msd = False, plot_dapp = False, plot_pie_chart = False):
        """Average tracks by model and optionally plot the results.
        This is the Minflux version. That means that the tracks potentially have very different lengths. 
        It makes use of the MF_model_average module heavily.

        Parameters
        ----------
        avg_only_params: bool
            Only average the model parameters but not D_app and MSD
        plot_msd: bool
            Plot the averaged MSD for each model.
        plot_dapp: bool
            Plot the averaged D_app for each model.
        plot_pie_chart: bool
            Plot a pie chart showing the relative fractions of each model.

        Returns
        -------
        results : dict
            Relative shares and averaged values of each model.
        """
        warnings.warn('MODEL UNDER CONSTRUCTION')
        
        if avg_only_params and (plot_msd or plot_dapp):
            warnings.warn("avg_only_params is True. plot_msd or plot_dapp will have no effect.")

        track_length = 0
        max_t = 0.0
        t = None
        for track in self._tracks:
            if track.get_t()[-1] > max_t:
                max_t = track.get_t()[-1]
                track_length = track.get_x().size
                t = track.get_t()

        average_D_app = {}
        average_MSD = {}
        average_params = {}
        sampled = {}
        counter = {}

        dt = t[1] - t[0]

        k = 0
        for track in self._tracks:
            k += 1
            if track.get_t()[1] - track.get_t()[0] != dt and not avg_only_params and not interpolation:
                raise ValueError("Cannot average MSD and D_app: Encountered track with incorrect time step size! "
                                 "(Got {}, expected {} for track {}.) Use the flag avg_only_params = True or "
                                 "enable interpolation with interpolation = True.".format(
                    track.get_t()[1] - track.get_t()[0], dt, k + 1))

        # Parameter averaging
        for track in self._tracks:
            if track.get_adc_analysis_results() is None:
                continue
            model = track.get_adc_analysis_results()["best_model"]
            if model is not None:
                if not model in average_params.keys():
                    average_params[model] = len(track.get_adc_analysis_results()["fit_results"][model]["params"]) * [0.0]
                    counter[model] = 0
                counter[model] += 1
                average_params[model] += track.get_adc_analysis_results()["fit_results"][model]["params"]
        
        for model in average_params.keys():
            average_params[model] /= counter[model]
        
        k = 0
        for track in self._tracks:
            k += 1
            if track.get_t()[1] - track.get_t()[0] != dt and not avg_only_params and not interpolation:
                raise ValueError("Cannot average MSD and D_app: Encountered track with incorrect time step size! "
                                 "(Got {}, expected {} for track {}.) Use the flag avg_only_params = True or "
                                 "enable interpolation with interpolation = True.".format(
                    track.get_t()[1] - track.get_t()[0], dt, k + 1))

        if not avg_only_params:
            for track in self._tracks:
                if track.get_adc_analysis_results() is None:
                    continue

                model = track.get_adc_analysis_results()["best_model"]

                D_app = np.zeros(track_length - 3)
                MSD = np.zeros(track_length - 3)
                if interpolation:
                    interp_MSD = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                    interp_D_app = interpolate.interp1d(track.get_t()[0:-3], track.get_adc_analysis_results()["Dapp"], bounds_error = False, fill_value = 0)
                    MSD = interp_MSD(t[0:-3])
                    D_app = interp_D_app(t[0:-3])
                else:
                    D_app[0:track._adc.size-3] = track.get_adc_analysis_results()["Dapp"][0:-3]
                    MSD[0:track._msd.size-3] = track.get_msd()[0:-3]
                mask = np.zeros(track_length - 3)
                np.put(mask, np.where(MSD != 0.0), 1)


                if not model in average_D_app.keys():
                    average_D_app[model] = np.zeros(track_length - 3)
                    average_MSD[model] = np.zeros(track_length - 3)
                    sampled[model] = np.zeros(track_length - 3)

                average_D_app[model] += D_app
                average_MSD[model] += MSD
                sampled[model] += mask

        counter_sum = 0
        for model in counter:
            counter_sum += counter[model]
            if counter[model]:
                average_D_app[model] /= sampled[model]
                average_MSD[model] /= sampled[model]

        if counter_sum == 0:
            warnings.warn("No tracks are categorized!")

        sector = {}
        for model in counter:
            sector[model] = counter[model] / len(self._tracks)
        sector["not catergorized"] = (len(self._tracks) - counter_sum) / len(self._tracks)

        if plot_msd and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average MSD")
            for model in counter:
                ax.semilogx(t[0:-3], average_MSD[model], label=model)
            ax.legend()

        if plot_dapp and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            min_val = 9999999.9
            max_val = 0.0
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average D_app")
            for model in counter:
                new_min = np.min(average_D_app[model])
                new_max = np.max(average_D_app[model])
                min_val = min(min_val, new_min)
                max_val = max(max_val, new_max)
                l, = ax.semilogx(t[0:-3], average_D_app[model], label=model)
                r = average_params[model]
                for c in ModelDB().models:
                    if c.__class__.__name__ == model:
                        m = c
                pred = m(t, *r)
                plt.semilogx(t[0:-3], pred[0:-3], linestyle='dashed', color=l.get_color())
            ax.set_ylim(0.95*min_val, 1.05*max_val)
            ax.legend()

        if plot_pie_chart:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.pie(sector.values(),
                   labels=sector.keys())

        return {"sectors": sector,
                "average_params": average_params, 
                "average_msd": average_MSD,
                "average_dapp": average_D_app}
    


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
    