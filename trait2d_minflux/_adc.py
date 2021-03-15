from trait2d.analysis import ModelDB

import numpy as np

def MF_adc_analysis(self, R: float = 1/6, fraction_fit_points: float=0.25, fit_max_time: float = None, maxfev = 1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
    """Revised analysis using the apparent diffusion coefficient

    Parameters
    ----------
    R: float
        Point scanning across the field of view.
    fraction_fit_points: float
        Fraction of points to use for fitting. Defaults to 25 %.
    fit_max_time: float
        Maximum time in fit range. Will override fraction_fit_points.
    maxfev: int
        Maximum function evaluations by scipy.optimize.curve_fit. The fit will fail if this number is exceeded.
    enable_log_sampling: bool
        Only sample logarithmically spaced time points for analysis.
    log_sampling_dist: float
        Exponent of logarithmic sampling (base 10).
    weighting: str
        Weighting of the datapoints used in the fit residual calculation. Can be `error` (weight by inverse standard
        deviation), `inverse_variance` (weight by inverse variance), `variance` (weight by variance)
        or `disabled` (no weighting). Default is `error`.

    Returns
    -------
    adc_analysis_results: dict
        Dictionary containing all analysis results.
        Can also be retreived using `Track.get_adc_analysis_results()`.
    """
        # Calculate MSD if this has not been done yet.
    if self._msd is None:
        self.MF_calculate_msd()
    
    dt = self._tn[0]
    N = self._msd.size
    
    #time array
    
    T = self._tn[0:N+1]
    
    #compute apparent diffusion coefficient arrays
    
    self._adc = Dapp = self._msd / (4 * T * (1-2*R*dt / T))
    self._adc_error = Dapp_err = self._msd_error / (4 * T * (1 - 2*R*dt / T))
    
    #do the fitting
    
    model, fit_indices, fit_results = self._MF_categorize(np.array(Dapp), np.arange(
        0, N), Dapp_err = Dapp_err, R=R, fraction_fit_points=fraction_fit_points, fit_max_time=fit_max_time, maxfev=maxfev, enable_log_sampling=enable_log_sampling, log_sampling_dist=log_sampling_dist, weighting = weighting)
    
    self._adc_analysis_results = {}
    self._adc_analysis_results["Dapp"] = np.array(Dapp)
    self._adc_analysis_results["Dapp_err"] = np.array(Dapp_err)
    self._adc_analysis_results["fit_indices"] = fit_indices
    self._adc_analysis_results["fit_results"] = fit_results
    self._adc_analysis_results["best_model"] = model

    return self._adc_analysis_results