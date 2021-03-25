import pickle as pkl
import pandas as pd
import os
import numpy as np
from scipy import optimize
import numpy as np
import warnings
import tqdm

from trait2d.analysis import ListOfTracks
from trait2d.analysis import Track

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class ModelDB(Borg):
    """Singleton class holding all models that should be used in analysis."""
    models = []
    def __init__(self):
        Borg.__init__(self)
    def add_model(self, model):
        """Add a new model class to the ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* an instance) to add. There are predefined models available in
            trait2d.analysis.models. Example usage:

            .. code-block:: python

                from trait2d.analysis.models import ModelConfined
                ModelDB().add_model(ModelConfined)
        """
        for m in self.models:
            if m.__class__ == model:
                raise ValueError("ModelDB already contains an instance of the model {}.".format(model.__name__))
        self.models.append(model())

    def get_model(self, model):
        """
        Return the model instance from ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* and instance) to remove. Example usage:

            .. code-block:: python
            
                from trait2d.analysis.models import ModelConfined
                ModelDB().get_model(ModelConfined).initial = [1.0e-12, 1.0e-9, 0.5e-3]
        """

        for i in range(len(self.models)):
            if model == self.models[i].__class__:
                return self.models[i]
        raise ValueError("ModelDB does not contain an instance of the model {}.".format(model.__name__))

    def remove_model(self, model):
        """
        Remove a model from ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* an instance) to remove. Example usage:

            .. code-block:: python

                from trait2d.analysis.models import ModelConfined
                ModelDB().remove_model(ModelConfined)
        """
        for i in range(len(self.models)):
            if model == self.models[i].__class__:
                self.models.pop(i)
                return
        raise ValueError("ModelDB does not contain an instance of the model {}.".format(model.__name__))

    def cleanup(self):
        """
        Remove all models from ModelDB. It is good practice to call this
        at the end of your scripts since other scripts might share the same
        instance.
        """
        self.models = []

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
        self._msd = np.empty(0)
        self._tn = np.empty(0)
        self._msd_error= np.empty(0)
        self._tn_error = np.empty(0)
        
        from tqdm import tqdm
        
        for i in tqdm(range(max(classification.labels_))):
            cluster_idx = np.where(classification.labels_==i) #select which of the unique_time_intervals are to be considered now
            #now we select the indices of the time_intervals matrix (lower triangular) fall into the cluster selected above
            idx = np.where((time_intervals>=np.min(unique_time_intervals[cluster_idx]))&(time_intervals<=np.max(unique_time_intervals[cluster_idx])))
            #the calculation of the MSD is now trivial 
            self._msd = np.append(self._msd,np.mean(sdis_matrix[idx]))
            self._msd_error = np.append(self._msd_error,np.std(sdis_matrix[idx])/np.sqrt(len(idx)))
            self._tn = np.append(self._tn,np.mean(tint_matrix_og[idx]))
            self._tn_error = np.append(self._tn_error,np.std(tint_matrix_og[idx])/np.sqrt(len(idx)))
        
    from ._msd import MF_msd_analysis
    from ._adc import MF_adc_analysis
            
    def _MF_categorize(self, Dapp, J, Dapp_err = None, R: float = 1/6, fraction_fit_points: float = 0.25, fit_max_time: float=None, maxfev=1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
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
            
        print(f'n_points ={n_points}')  
        
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
            
        print(idxs)
        
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
    