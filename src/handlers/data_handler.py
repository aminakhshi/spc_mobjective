import os
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import namedtuple
import joblib
import yaml
import neo.io  
from utils.feature_extractor import SpikeFeatureExtractor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
cm = 1/2.54

def update_props(ticklabel_size=12, legend_size=10, label_size=12):
    """
    Update the font sizes of tick labels, legends, and labels in Matplotlib.

    Parameters:
        ticklabel_size (int): Font size for tick labels. Default is 10.
        legend_size (int): Font size for legends. Default is 10.
        label_size (int): Font size for labels. Default is 12.
    """
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.rc('xtick', labelsize=ticklabel_size)
    plt.rc('ytick', labelsize=ticklabel_size)
    plt.rc('legend', fontsize=legend_size)
    plt.rc('axes', labelsize=label_size)

update_props()


logger = logging.getLogger(__name__)
Solution = namedtuple('Solution', ['t', 'y'])

def convert_to_native(obj):
    """
    Recursively convert NumPy data types to native Python types.
    
    Parameters:
    -----------
    obj : Any
        The object to convert.
    
    Returns:
    --------
    Any
        The converted object with native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def separate_spikes(spiketimes, burst_th, **kwargs):
    # cut_off = kwargs.get('cut_off', False) 
    isis = np.diff(spiketimes)  # compute the ISI sequence
    isis = isis[isis >= 0.001]  # remove ISI values less than 0.001
    # threshold = burst_th  # set the threshold to 10 msec, change as necessary
    burst_spiketimes = []
    burst_events = []
    isolated_spiketimes = spiketimes.copy()  # initially all spikes are isolated
    if len(spiketimes) > 2:
        current_sequence = [spiketimes[0]]
        for i, val in enumerate(isis):
            if val < burst_th:
                current_sequence.append(spiketimes[i+1])
            else:
                if len(current_sequence) > 1:
                    burst_events.append(current_sequence)
                    burst_spiketimes.extend(current_sequence)
                # else:
                    # isolated_spiketimes.append(current_sequence[0])
                current_sequence = [spiketimes[i+1]]
        if current_sequence:
            if len(current_sequence) > 1:
                burst_events.append(current_sequence)
                burst_spiketimes.extend(current_sequence)
            # else:
                # isolated_spiketimes.append(current_sequence[0])
        
        burst_spiketimes = np.array(burst_spiketimes)
        isolated_spiketimes = np.setdiff1d(isolated_spiketimes, burst_spiketimes)
    return burst_spiketimes, isolated_spiketimes, burst_events
        

class DataHandler:
    def __init__(self, data: dict, params: dict, config: dict, run_id: int, 
                 save_dir: str = "results", 
                 isi_plot_type: str = 'histogram', **kwargs):
        """
        Initialize with data dictionary containing 't' and 'y', along with parameters and config.

        Parameters:
        -----------
        data : dict
            Dictionary containing 't' (time points) and 'y' (solution array).
        params : dict
            Dictionary of model parameters used in the simulation.
        config : dict
            Dictionary of simulation configurations.
        run_id : int
            Unique identifier for the simulation run.
        save_dir : str, optional
            Directory where results will be saved. Defaults to "results".
        isi_plot_type : str, optional
            Type of ISI plot: 'histogram' or 'kde'. Defaults to 'histogram'.
        **kwargs : dict
            Additional keyword arguments:
                - cutoff (float): Fraction of data to remove as transient. Defaults to 0.25.
                - save_result (bool): Whether to save the results. Defaults to True.
                - threshold (float): Threshold for spike detection. Defaults to 0.0.
                - distance (int): Minimum distance between peaks in samples. Defaults to 20.
        """
        self.t = data.get('t', np.array([]))
        self.y = data.get('y', np.array([]))
        self.spike_times = None
        self.isi = None
        self.isi_stats = {}
        self.cutoff = kwargs.pop('cutoff', 0.25)
        self.save_result = kwargs.pop('save_result', True)
        
        self.params = params
        config.pop('t_eval', None)
        self.config = config
        self.run_id = run_id
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists early
        self.isi_plot_type = isi_plot_type.lower()

        logger.info("Initializing DataHandler for run_id=%d with cutoff=%.2f, save_result=%s, isi_plot_type='%s'",
                    self.run_id, self.cutoff, self.save_result, self.isi_plot_type)

        # Validate data shapes
        if self.y.ndim != 2:
            logger.error("Data 'y' must be a 2D array with shape (n_vars, n_times). Got shape: %s", self.y.shape)
            raise ValueError("Data 'y' must be a 2D array with shape (n_vars, n_times).")

        if self.t.ndim != 1:
            logger.error("Data 't' must be a 1D array. Got shape: %s", self.t.shape)
            raise ValueError("Data 't' must be a 1D array.")

        if isinstance(self.cutoff, (int, float)) and self.cutoff > 0:
            logger.info("Removing %.2f fraction of data to eliminate transients.", self.cutoff)
            cutoff_ind = int(len(self.t) * self.cutoff)
            self.t = self.t[cutoff_ind:] - self.t[cutoff_ind]
            self.y = self.y[:, cutoff_ind:]  
            logger.debug("Data after cutoff: t.shape=%s, y.shape=%s", self.t.shape, self.y.shape)
        elif isinstance(self.cutoff, (tuple, list)):
            start_cut, end_cut = self.cutoff
            cutoff_ind = np.where((self.t > start_cut) & (self.t <= end_cut))[0]
            self.t = self.t[cutoff_ind] 
            self.t = self.t - self.t[0]
            self.y = self.y[:, cutoff_ind]
            logger.debug("Data after cutoff: t.shape=%s, y.shape=%s", self.t.shape, self.y.shape)

        if self.save_result:
            self.spike_times, self.isi = self.get_spikes(**kwargs)
            self.isi_stats = self.get_isi_features()
            self.instant_freq, self.inst_freq_times = self.get_inst_freq()
            self.plot_spikes()
            self.save_additional_data()

    def get_spikes(self, threshold=0.0, distance=20):
        """
        Detect spikes in the data.

        Parameters:
        -----------
        threshold : float, optional
            Minimum height of peaks. Defaults to 0.0.
        distance : int, optional
            Minimum number of samples between peaks. Defaults to 20.

        Returns:
        --------
        tuple
            spike_times (np.ndarray): Times at which spikes occur.
            isi (np.ndarray): Inter-Spike Intervals.
        """
        logger.info("Detecting spikes for run_id=%d with threshold=%.2f and distance=%d samples.", 
                    self.run_id, threshold, distance)
        try:
            peaks, properties = find_peaks(self.y[0], height=threshold, distance=distance)
            spike_times = self.t[peaks]
            isi = np.diff(spike_times)
            logger.info("Run_id=%d: Number of spikes detected: %d", self.run_id, len(spike_times))
            logger.debug("Run_id=%d: Spike times: %s", self.run_id, spike_times)
            logger.debug("Run_id=%d: Inter-Spike Intervals (ISI): %s", self.run_id, isi)
            return spike_times, isi
        except Exception as e:
            logger.error("Run_id=%d: Error in spike detection: %s", self.run_id, e)
            raise RuntimeError(f"Error in spike detection for run_id={self.run_id}: {e}") from e

    def get_isi_features(self):
        """
        Compute ISI statistics.

        Returns:
        --------
        dict
            Dictionary containing ISI statistics.
        """
        logger.info("Computing ISI statistics for run_id=%d.", self.run_id)
        if self.isi is None:
            logger.error("Run_id=%d: ISI not computed. Call get_spikes() first.", self.run_id)
            raise RuntimeError("ISI not computed. Call get_spikes() first.")

        if len(self.isi) == 0:
            logger.warning("Run_id=%d: No ISI data to compute statistics.", self.run_id)
            return {}

        mean_isi = float(np.mean(self.isi))
        median_isi = float(np.median(self.isi))
        quantiles_isi = [float(q) for q in np.quantile(self.isi, [0.25, 0.75])]
        std_isi = float(np.std(self.isi))
        var_isi = float(np.var(self.isi))
        cv = float(std_isi / mean_isi) if mean_isi != 0 else float('nan')
        fano_factor = float(var_isi / mean_isi) if mean_isi != 0 else float('nan')

        self.isi_stats = {
            'mean': mean_isi,
            'median': median_isi,
            'quantiles': quantiles_isi,  # list of floats
            'std': std_isi,
            'cv': cv,
            'fano_factor': fano_factor,
            'spike_time': self.spike_times
        }

        logger.info("Run_id=%d: ISI Statistics: %s", self.run_id, self.isi_stats)
        logger.debug(
            "Run_id=%d: ISI Mean: %.6f, Median: %.6f, Quantiles: %s, Std: %.6f, CV: %.6f, Fano Factor: %.6f",
            self.run_id, mean_isi, median_isi, quantiles_isi, std_isi, cv, fano_factor
        )
        return self.isi_stats

    def get_inst_freq(self):
        """
        Computes the instantaneous frequency based on ISI.

        Returns:
            tuple: (instantaneous_freq (np.ndarray), corresponding_times (np.ndarray))
        """
        logger.info("Computing instantaneous frequency for run_id=%d.", self.run_id)
        if self.spike_times is None:
            logger.error("Run_id=%d: Spike times not available. Call get_spikes() first.", self.run_id)
            raise RuntimeError("Spike times not available. Call get_spikes() first.")

        if len(self.spike_times) < 2:
            logger.warning("Run_id=%d: Not enough spikes to compute instantaneous frequency.", self.run_id)
            return np.array([]), np.array([])

        try:
            isi = np.diff(self.spike_times)  # ISI in the same time units as self.t
            # Avoid division by zero
            # isi = np.where(isi == 0, np.nan, isi)
            instantaneous_freq = 1.0 / isi  # Frequency in Hz if self.t is in seconds
            freq_times = self.spike_times[1:]  # Assign frequency to the second spike in each pair

            self.instant_freq = instantaneous_freq
            self.inst_freq_times = freq_times

            logger.info("Run_id=%d: Computed instantaneous frequency for %d spikes.", self.run_id, len(instantaneous_freq))
            logger.debug("Run_id=%d: Instantaneous Frequency: %s", self.run_id, instantaneous_freq)
            logger.debug("Run_id=%d: Frequency Times: %s", self.run_id, freq_times)

            return instantaneous_freq, freq_times
        except Exception as e:
            logger.error("Run_id=%d: Error in computing instantaneous frequency: %s", self.run_id, e)
            raise RuntimeError(f"Error in computing instantaneous frequency for run_id={self.run_id}: {e}") from e

    def plot_spikes(self, t_min = None, t_max = None):
       """
       Create and save the simulation plot based on the simulation data.
       """
       logger.debug("Creating plot for run_id=%d.", self.run_id)
       try:
           # Define ISI plot type: 'histogram' or 'kde'
           isi_plot_type = self.isi_plot_type
           # ---------------------------
           # Define the Time Window
           # ---------------------------
           t_min = self.t[0] if t_min is None else t_min
           t_max = self.t[-1] if t_max is None else t_max

           # Create a figure with five subplots using gridspec
           fig = plt.figure(figsize=(15*cm, 12*cm), constrained_layout=True)
           gs = fig.add_gridspec(4, 2, height_ratios=[0.2, 0.1, 0.4, 0.3],
                                 hspace=0.05, width_ratios=[0.85, 0.15], wspace=0.06)  # Total height ratios sum to 12

           # ---------------------------
           # Subplot 1: Instantaneous Frequency
           # ---------------------------
           ax1 = fig.add_subplot(gs[0, 0])
           # ax1.plot(inst_freq_times_new, instant_freq_new, color='navy', label='Instantaneous Frequency')
           ax1.plot(self.inst_freq_times, self.instant_freq, color='navy', label='Inst Freq')
           ax1.axhline(50, color='orange', ls='-.', linewidth=0.8, label='50 Hz')
           ax1.axhline(100, color='red', ls='-.', linewidth=0.8, label='100 Hz')
           ax1.axis('off')  
           ax1.legend(loc='upper right')

           # ---------------------------
           # Subplot 2: Spike Times
           # ---------------------------
           ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
           # ax2.vlines(spike_times_new, 0, 1, color='black', linewidth=0.8)
           ax2.vlines(self.spike_times, 0, 1, color='black', linewidth=0.8)
           ax2.set_ylim(0, 1)
           ax2.axis('off')  

           # ---------------------------
           # Subplot 3: Membrane Potential x
           # ---------------------------
           ax3 = fig.add_subplot(gs[2, 0])
           ax3.plot(self.t, self.y[0], label='Membrane Potential x', color='royalblue')
           ax3.set_ylabel('v_x')
           ax3.legend()
           ax3.spines[['right', 'top']].set_visible(False)
           ax3.get_xaxis().set_visible(False)
           
           # ---------------------------
           # Subplot 4: z Variable
           # ---------------------------
           ax4 = fig.add_subplot(gs[3, 0], sharex=ax3)
           ax4.plot(self.t, self.y[2], label='z Variable', color='orangered')
           ax4.plot(self.t, self.y[-1], label='act', color='orange')
           ax4.set_xlabel('Time (ms)')
           ax4.set_ylabel('z_adp')
           ax4.legend()
           ax4.spines[['right', 'top']].set_visible(False)

           # ---------------------------
           # Subplot 5: ISI Histogram or KDE
           # ---------------------------
           ax5 = fig.add_subplot(gs[2:, 1])
           ax5.spines[['right', 'top']].set_visible(False)
           if isi_plot_type == 'histogram':
               bins = int(len(self.isi)/5) if len(self.isi) > 25 else 11
               sns.histplot(y=self.isi, bins=bins, kde=False, 
                            element="step", fill=True, 
                            color='dodgerblue', ax=ax5)

           elif isi_plot_type == 'kde':
               sns.kdeplot(y=self.isi, fill=True, color='dodgerblue', ax=ax5)

           else:
               logger.warning("Unknown ISI plot type: '%s'. Defaulting to histogram.", isi_plot_type)
               sns.histplot(y=self.isi, bins=bins, kde=False, 
                            element="step", fill=True, 
                            color='dodgerblue', ax=ax5)
               ax5.set_title('ISI Histogram')
               ax5.set_xlabel('ISI (ms)')
               ax5.set_ylabel('Count')
           ax5.grid(True)
           
           axes_to_limit = [ax1, ax2, ax3, ax4]
           for ax in axes_to_limit:
               ax.set_xlim(t_min, t_max)
           
           if self.save_result:
               plot_path = self.save_dir / f"plot_run_{self.run_id}.pdf"
               # plt.savefig(os.path.join(args.save_dir, f'kde_stats.pdf'), dpi=300, bbox_inches='tight')
               plt.savefig(plot_path, dpi=300, bbox_inches='tight')
               logger.info("Run_id=%d: Saved plot to %s", self.run_id, plot_path)
               
           plt.close(fig)
           logger.debug("Run_id=%d: Plot created and saved.", self.run_id)
       except Exception as e:
           logger.error("Run_id=%d: Error in creating/saving plot: %s", self.run_id, e)

    def save_additional_data(self):
        """
        Save simulation parameters and ISI statistics individually.
        Also, save the configuration once.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration once
        config_path = self.save_dir / "config.yaml"
        if not config_path.exists():
            try:
                config_native = convert_to_native(self.config)
                if isinstance(config_native, dict) and 't_eval' in config_native:
                    config_native.pop('t_eval')
                with open(config_path, 'w') as f:
                    yaml.dump(config_native, f)
                logger.info("Saved simulation configurations to %s", config_path)
            except Exception as e:
                logger.error("Run_id=%d: Error saving configuration: %s", self.run_id, e)
        else:
            logger.debug("Configuration file %s already exists. Skipping save.", config_path)

        # Save run-specific data
        try:
            # Define file paths
            params_path = self.save_dir / f"params_run_{self.run_id}.yaml"

            params_native = convert_to_native(self.params)
            isi_stats_native = convert_to_native(self.isi_stats)
            # Save parameters and spike statistics
            with open(params_path, 'w') as f:
                yaml.safe_dump({'params': params_native,
                                'isi_stats': isi_stats_native,
                                'run_id': self.run_id,}, f)
            logger.debug("Run_id=%d: Saved output stats and parameters to %s", self.run_id, params_path)

        except Exception as e:
            logger.error("Run_id=%d: Error saving run data: %s", self.run_id, e)
            raise RuntimeError(f"Error saving run data for run_id={self.run_id}: {e}") from e

class DataHandlerOptim:
    """
    The data handler class for neural data processing.
    It supports membrane potential recordings from Spike2, spike train data, and model simulation data.

    Depending on the input data type, it performs spike feature extraction, spike train feature extraction,
    or both, and provides functionalities for saving the results.
    """

    def __init__(
        self,
        data: Optional[Union[Dict[str, np.ndarray], str, namedtuple]] = None,
        data_type: Optional[str] = None,  # 'recording', 'spiketrain', 'simulation'
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        save_result: bool = True,
        **kwargs
    ):
        """
        Initialize the DataHandler with data, parameters, and configurations.

        Parameters:
        -----------
        data : dict, str, or namedtuple, optional
            - If dict: Should contain 't' (time points) and 'y' (solution array) for simulation data
            - If str: File path to recording data (e.g., Spike2 .smr files)
            - If namedtuple: Simulation data with fields 't' and 'y'
        data_type : str, optional
            Type of the input data. Must be one of ['recording', 'spiketrain', 'simulation'].
            If not provided, the type is inferred from the 'data' input.
        params : dict, optional
            Dictionary of model parameters used in the simulation.
        config : dict, optional
            Dictionary of simulation configurations.
        save_result: bool, 
            Determines if the result is saved. Defaults to True
        **kwargs : dict
            Additional keyword arguments:
                - features (str): Type of features to be extracted. For 'recording' or 'simulation' data types defaults to "all", for 'spiketrain' defaults to 'isi'
                - save_dir (str): Directory where results will be saved. Defaults to "results".
                - run_id (int): Unique identifier for the simulation run.
                - cutoff (float or tuple/list): Fraction of data to remove as transient. Defaults to 0.25.
                - threshold (float): Threshold for spike detection. Defaults to 0.0.
                - distance (int): Minimum distance between peaks in samples. Defaults to 20.
        """
        self.params = params if params else {}
        self.config = config if config else {}
        self.save_result = save_result
        self.run_id = kwargs.get('run_id', 0)
        
        # Determine data type
        if data_type:
            if data_type.lower() not in ['recording', 'spiketrain', 'simulation']:
                logger.error("Invalid data_type '%s'. Must be one of ['recording', 'spiketrain', 'simulation'].", data_type)
                raise ValueError("data_type must be one of ['recording', 'spiketrain', 'simulation'].")
            self.data_type = data_type.lower()
        else:
            self.data_type = self._get_data_type(data)
            logger.info("Data type inferred as '%s'.", self.data_type)
        
        # Initialize data attributes
        self.t: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.spike_times: Optional[Union[List[np.ndarray], np.ndarray]] = None
        self.bin_spikes: Optional[np.ndarray] = None
        # Initialize feature attributes
        self.spike_features: Optional[pd.DataFrame] = None
        self.spike_train_features: Optional[pd.DataFrame] = None
        
        self._load_data(data)    

        # Validate data shapes if applicable
        if self.data_type in ['recording', 'simulation']:
            if self.y.ndim == 1:
                self.y = self.y.reshape(1, -1)
                logger.debug("Reshaped 'y' to shape: %s", self.y.shape)
            elif self.y.ndim != 2:
                logger.error("Data 'y' must be a 2D array with shape (n_vars, n_times). Got shape: %s", self.y.shape)
                raise ValueError("Data 'y' must be a 2D array with shape (n_vars, n_times).")
            
            if self.y.shape[-1] != len(self.t):
                logger.error("Data 'y' must have the same length as 't'. Got shape y: %s, shape t: %s", self.y.shape, len(self.t))
                raise ValueError("Data 'y' must have the same length as 't'.")

            if self.t.ndim != 1:
                logger.error("Data 't' must be a 1D array. Got shape: %s", self.t.shape)
                raise ValueError("Data 't' must be a 1D array.")
            
            if 'cutoff' not in self.config:
                self.config['cutoff'] = kwargs.get('cutoff', 0.0)  
            if isinstance(self.config['cutoff'], (int, float)) and self.config['cutoff'] > 0:
                logger.info("Removing %.2f fraction of data to eliminate transients.", self.config['cutoff'])
                cutoff_ind = int(len(self.t) * self.config['cutoff'])
                self.t = self.t[cutoff_ind:] - self.t[cutoff_ind]
                self.y = self.y[:, cutoff_ind:]  
                logger.debug("Data after cutoff: t.shape=%s, y.shape=%s", self.t.shape, self.y.shape)
            elif isinstance(self.config['cutoff'], (tuple, list)):
                start_cut, end_cut = self.config['cutoff']
                cutoff_ind = np.where((self.t > start_cut) & (self.t <= end_cut))[0]
                self.t = self.t[cutoff_ind] 
                self.t = self.t - self.t[0]
                self.y = self.y[:, cutoff_ind]
                logger.debug("Data after cutoff: t.shape=%s, y.shape=%s", self.t.shape, self.y.shape)

            self.features = kwargs.get('features', 'all')
            self.config['t_start'] = kwargs.get('t_start', self.t[0])
            self.config['t_end'] = kwargs.get('t_end', self.t[-1])
            if 'duration' not in self.config:
                self.config['duration'] = float(self.config['t_end']-self.config['t_start'])

            if 'fs' not in self.config:
                self.config['fs'] = kwargs.get('fs', 1.0)  # in Hz
            
            self.config['sta_win'] = kwargs.get('sta_win', 10.0)  # in milliseconds
            self.config['spkt_ref'] = kwargs.get('spkt_ref', 'peak_index')  # reference for spike times
            if self.data_type == 'recording':
                self.config['dv_cutoff'] = kwargs.get('dv_cutoff', 20.0)
                self.config['min_height'] = kwargs.get('min_height', 2.0)
                self.config['min_peak'] = kwargs.get('min_peak', -30.0)
            else:  # 'simulation'
                self.config['dv_cutoff'] = kwargs.get('dv_cutoff', 2.0)
                self.config['min_height'] = kwargs.get('min_height', 1.0)
                self.config['min_peak'] = kwargs.get('min_peak', -1.0)
            self.config['thresh_frac'] = kwargs.get('thresh_frac', 0.05)
            self.config['max_interval'] = kwargs.get('max_interval', 0.005)
            self.config['filter'] = kwargs.get('filter', 5.0)
     
            if self.features in ['all', 'spike']:
                self.extract_features(time=self.t,
                                      data=self.y,
                                      feature_type=self.features,
                                      **self.config)
    
        elif self.data_type == 'spiketrain':
            self.features = 'isi'  # only extract spike train features
            self.extract_features(spike_times=self.spike_times,
                                  feature_type=self.features,
                                  **self.config)

        # If save_result is True, proceed with saving feature data
        if self.save_result:
            self.save_dir  = kwargs.get('save_dir', 'results')
            self.save_dir = Path(self.save_dir).resolve()
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_additional_data()


    def _get_data_type(self, data: Union[Dict[str, np.ndarray], str, namedtuple, None]) -> str:
        """
        Infer the data type based on the input data.

        Parameters:
        -----------
        data : dict, str, namedtuple, or None
            Input data.

        Returns:
        --------
        str
            Inferred data type: 'recording', 'spiketrain', or 'simulation'.
        """
        if isinstance(data, str):
            logger.debug("Data is a string. Assuming 'recording' type.")
            return 'recording'
        elif isinstance(data, dict):
            if 'spike_times' in data:
                logger.debug("Data dictionary contains 'spike_times'. Assuming 'spiketrain' type.")
                return 'spiketrain'
            elif 't' in data and 'y' in data:
                logger.debug("Data dictionary contains 't' and 'y'. Assuming 'simulation' type.")
                return 'simulation'
            else:
                logger.error("Data dictionary must contain either 't' and 'y' for simulation or 'spike_times' for spike trains.")
                raise ValueError("Invalid data dictionary. Must contain 't' and 'y' for simulation or 'spike_times' for spike trains.")
        elif isinstance(data, (tuple, namedtuple)):
            if hasattr(data, 't') and hasattr(data, 'y'):
                logger.debug("Data is a namedtuple with 't' and 'y'. Assuming 'simulation' type.")
                return 'simulation'
            else:
                logger.error("Namedtuple must have 't' and 'y' attributes for simulation data.")
                raise ValueError("Invalid namedtuple. Must have 't' and 'y' attributes for simulation data.")
        elif data is None:
            logger.error("No data provided.")
            raise ValueError("No data provided.")
        else:
            logger.error("Unsupported data type: %s", type(data))
            raise TypeError("Unsupported data type.")

    def _load_data(self, data: Union[Dict[str, np.ndarray], str, namedtuple, None]):
        """
        Load and parse the input data based on its type.

        Parameters:
        -----------
        data : dict, str, namedtuple, or None
            Input data.
        """
        if self.data_type == 'recording':
            if not isinstance(data, str):
                logger.error("For 'recording' data_type, 'data' must be a file path string.")
                raise TypeError("For 'recording' data_type, 'data' must be a file path string.")
            self._load_recording(data)
        elif self.data_type == 'spiketrain':
            if not isinstance(data, dict):
                logger.error("For 'spiketrain' data_type, 'data' must be a dictionary containing 'spike_times' or 'bin_spikes'.")
                raise TypeError("For 'spiketrain' data_type, 'data' must be a dictionary containing 'spike_times' or 'bin_spikes'.")
            self._load_spiketrain(data)
        elif self.data_type == 'simulation':
            if isinstance(data, dict):
                self.t = data.get('t', np.array([]))
                self.y = data.get('y', np.array([]))
            elif isinstance(data, tuple) or isinstance(data, namedtuple):
                self.t = np.array(data.t)
                self.y = np.array(data.y)
            else:
                logger.error("For 'simulation' data_type, 'data' must be a dictionary or namedtuple with 't' and 'y'.")
                raise TypeError("For 'simulation' data_type, 'data' must be a dictionary or namedtuple with 't' and 'y'.")
        else:
            logger.error("Unsupported data type: %s", self.data_type)
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def _load_recording(self, file_path: str):
        """
        Load membrane potential recordings from a Spike2 .smr file.

        Parameters:
        -----------
        file_path : str
            Path to the .smr recording file.
        """
        logger.info(f"Loading recording data from Spike2 file: {file_path}")
        if not os.path.isfile(file_path):
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                reader = neo.io.Spike2IO(filename=file_path)
                logger.info("Successfully initialized Spike2IO for file: %s", file_path)
            except Exception as e:
                logger.warning("Failed to initialize Spike2IO with default parameters: %s. Trying alternative settings.", e)
                reader = neo.io.Spike2IO(filename=file_path, try_signal_grouping=False)

            block = reader.read_block(lazy=False)

        # Assuming single segment for simplicity; extend as needed
        if len(block.segments) == 0:
            logger.error("No segments found in the Spike2 file: %s", file_path)
            raise ValueError(f"No segments found in the Spike2 file: {file_path}")

        segment = block.segments[0]

        # Extract analog signals (membrane potential)
        analog_signals = segment.analogsignals
        if not analog_signals:
            logger.error("No analog signals found in the Spike2 file: %s", file_path)
            raise ValueError(f"No analog signals found in the Spike2 file: {file_path}")
        
        # For simplicity, use the first analog signal that likely represents membrane potential
        for idx, signal in enumerate(analog_signals):
            if "V" in str(signal.units) and np.squeeze(signal.magnitude).ndim == 1:
                self.y = np.array(signal.magnitude).squeeze()
                self.t = np.array(signal.times.rescale('s').magnitude).astype(self.y.dtype)
                self.config['fs'] = float(signal.sampling_rate)
                self.config['unit'] = str(signal.units)
                self.config['t_span'] = (float(signal.t_start), float(signal.t_stop))
                break  # Exit after finding the first relevant signal

        if self.y.size == 0:
            logger.error("No suitable analog signal found for membrane potential in file: %s", file_path)
            raise ValueError(f"No suitable analog signal found for membrane potential in file: {file_path}")
        
        if self.y.ndim == 1:
            self.y = self.y.reshape(1, -1)
        
        logger.info("Loaded recording data: t.shape=%s, y.shape=%s", self.t.shape, self.y.shape)

    def _load_spiketrain(self, data: Dict[str, Any]):
        """
        Load spike train data.

        Parameters:
        -----------
        data : dict
            Dictionary containing 'spike_times' or 'bin_spikes', and 'fs'.
            Optionally, 'neuron_id' for multiple neurons.
        """
        logger.info("Loading spike train data.")
        required_keys = ['spike_times', 'bin_spikes']
        if not any(key in data for key in required_keys):
            logger.error("Spike train data must contain at least one of the required keys: %s.", required_keys)
            raise KeyError(f"Spike train data must contain at least one of {required_keys}.")
        
        self.config['fs'] = data.get('fs', 1.0)  # Default to 1 Hz if not provided
        fs = self.config['fs'] 

        if 'spike_times' in data:
            self.spike_train = data['spike_times']
            # Validate spike_train format
            if isinstance(self.spike_train, list):
                logger.info("Multiple spike trains detected: %d neurons.", len(self.spike_train))
                self.spike_train = [np.array(st) for st in self.spike_train]
            elif isinstance(self.spike_train, np.ndarray):
                logger.info("Single spike train detected.")
                self.spike_train = [self.spike_train]
            else:
                logger.error("Invalid format for 'spike_times'. Must be a list or numpy array.")
                raise TypeError("Invalid format for 'spike_times'. Must be a list or numpy array.")
            
            if 'bin_spikes' not in data:
                logger.debug("Computing 'bin_spikes' from 'spike_times'.")
                self.bin_spikes = self.get_bin_spikes(self.spike_train, fs)
            else:
                self.bin_spikes = data['bin_spikes']
                self._validate_bin_spikes(self.bin_spikes)
        else:
            self.bin_spikes = data['bin_spikes']
            self._validate_bin_spikes(self.bin_spikes)
            logger.debug("Computing 'spike_times' from 'bin_spikes'.")
            self.spike_train = self._get_spike_times(self.bin_spikes, fs)

        logger.info("Spike train data loaded successfully.")

    def _validate_bin_spikes(self, bin_spikes: Union[List[np.ndarray], np.ndarray]):
        """
        Validate the format of bin_spikes.

        Parameters:
        -----------
        bin_spikes : list or numpy array
            Binned spike data.
        """
        if isinstance(bin_spikes, list):
            logger.debug("Multiple binned spike trains detected: %d neurons.", len(bin_spikes))
            self.bin_spikes = [np.array(bs) for bs in bin_spikes]
        elif isinstance(bin_spikes, np.ndarray) and bin_spikes.ndim == 1:
            logger.info("Single binned spike train detected.")
            self.bin_spikes = [bin_spikes]
        else:
            logger.error("Invalid format for 'bin_spikes'. Must be a list or 1D numpy array.")
            raise TypeError("Invalid format for 'bin_spikes'. Must be a list or 1D numpy array.")

    def get_bin_spikes(self, spike_trains: List[np.ndarray], fs: float) -> List[np.ndarray]:
        """
        Convert spike times to binned spikes.

        Parameters:
        -----------
        spike_trains : list of numpy arrays
            List containing spike times for each neuron.
        fs : float
            Sampling frequency in Hz.

        Returns:
        --------
        List of numpy arrays representing binned spikes.
        """
        bin_size = 1.0 / fs  # Duration of each bin in seconds
        # Determine the maximum spike time to set the number of bins
        max_time = max(st.max() if st.size > 0 else 0 for st in spike_trains)
        num_bins = int(np.ceil(max_time / bin_size)) + 1

        binned_spikes = []
        for idx, st in enumerate(spike_trains):
            if st.size == 0:
                logger.warning(f"Neuron {idx+1} has no spikes.")
                binned = np.zeros(num_bins, dtype=int)
            else:
                binned, _ = np.histogram(st, bins=np.linspace(0, bin_size*num_bins, num_bins+1))
            binned_spikes.append(binned)
            logger.debug(f"Neuron {idx+1}: Binned spikes computed.")
        return binned_spikes

    def _get_spike_times(self, bin_spikes: List[np.ndarray], fs: float) -> List[np.ndarray]:
        """
        Convert binned spikes to spike times.

        Parameters:
        -----------
        bin_spikes : list of numpy arrays
            List containing binned spikes for each neuron.
        fs : float
            Sampling frequency in Hz.

        Returns:
        --------
        List of numpy arrays representing spike times.
        """
        bin_size = 1.0 / fs  # Duration of each bin in seconds
        spike_trains = []
        for idx, bs in enumerate(bin_spikes):
            spike_indices = np.where(bs > 0)[0]
            if spike_indices.size == 0:
                logger.warning(f"Neuron {idx+1} has no spikes.")
                spike_trains.append(np.array([]))
                continue
            # Assuming one spike per bin, assign spike time to the center of the bin
            spike_times = spike_indices * bin_size + bin_size / 2.0
            spike_trains.append(spike_times)
            logger.debug(f"Neuron {idx+1}: Spike times computed from binned spikes.")
        return spike_trains

    def get_spikes_from_spiketrain(self) -> List[np.ndarray]:
        """
        Retrieve spike times from spike train data.

        Returns:
        --------
        list of np.ndarray
            List containing spike times for each neuron.
        """
        logger.info("Retrieving spike times from spike train data for run_id=%d.", self.run_id)
        if not hasattr(self, 'spike_train') or not self.spike_train:
            logger.error("No spike train data available.")
            raise ValueError("No spike train data available.")

        spike_times_list = self.spike_train  # List of np.ndarray
        for idx, spike_times in enumerate(spike_times_list):
            logger.info("Neuron %d: %d spikes detected.", idx+1, len(spike_times))
            logger.debug("Neuron %d: Spike times: %s", idx+1, spike_times)
        return spike_times_list

    def extract_features(
        self,
        time: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        spike_times: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        bin_spikes: Optional[np.ndarray] = None,
        fs: float = 10000.0,
        feature_type: str = 'all',  # 'spike', 'isi', 'all'
        **kwargs
    ):
        """
        Perform feature extraction based on the data. For membrane potential data (e.g., simulation or recording) it supports spike features and/or spike train features; For spike train data it only supports spike train feature extraction.

        Parameters:
        -----------
        time : ndarray, optional
            Array of time points in seconds.
        data : ndarray, optional
            Array of data, e.g., membrane potential.
        spike_times : list of ndarray or ndarray, optional
            Array of spike times (for spike train data).
        bin_spikes : ndarray, optional
            Binned spike data.
        fs : float, optional
            Sampling frequency in Hz. Defaults to 10000.0.
        feature_type : str, optional
            Type of features to extract: 'spike', 'isi', or 'all'. Defaults to 'all'.
        **kwargs : dict
            Additional keyword arguments.
        """
        logger.info("Starting feature extraction for run_id=%d with feature_type='%s'.", self.run_id, feature_type)
        if feature_type not in ['all', 'spike', 'isi']:
            logger.error("Invalid feature_type '%s'. Must be one of ['all', 'spike', 'isi'].", feature_type)
            raise ValueError("feature_type must be one of ['all', 'spike', 'isi'].")
        elif self.data_type == 'spiketrain' and feature_type in ['all', 'spike']:
            logger.error("Invalid feature_type '%s' for spiketrain data. Change feature_type to 'isi'.", feature_type)
            raise ValueError("feature_type must be 'isi' for spiketrain data.")
        
        if self.data_type in ['recording', 'simulation'] and feature_type in ['spike', 'all']:
            try:
                logger.debug("Extracting spike features using SpikeFeatureExtractor.")
                stim = kwargs.get('stim', None)  # Assuming 'stim' is provided if applicable
                v = data[0]  # Assuming first row is voltage
                sfx = SpikeFeatureExtractor(
                    start=self.config['t_start'],
                    end=self.config['t_end'],
                    filter=self.config['filter'],
                    dv_cutoff=self.config['dv_cutoff'],
                    max_interval=self.config['max_interval'],
                    min_height=self.config['min_height'],
                    min_peak=self.config['min_peak'],
                    thresh_frac=self.config['thresh_frac']
                )
                spikes_df = sfx.process(t=time, v=v, i=stim)
                self.spike_features = spikes_df
                if spikes_df is not None and not spikes_df.empty:
                    if 'threshold_t' in spikes_df.columns:
                        self.spike_times = spikes_df['threshold_t'].values
                    elif 'peak_t' in spikes_df.columns:
                        self.spike_times = spikes_df['peak_t'].values
                    else:
                        logger.warning("Spike DataFrame does not contain 'threshold_t' or 'peak_t'.")
                        self.spike_times = spikes_df.iloc[:, 0].values  # Fallback to first column
                    logger.debug("Spike features extracted. Number of spikes: %d", len(self.spike_times))
                
                else:
                    # Handle the case when no spikes are detected
                    logger.warning(f"Run_id={self.run_id}: No spikes detected.")
                    self.spike_times = None
        
                if feature_type == 'all':
                    logger.debug("Extracting spike train features using internal methods.")
                    self.spike_train_features = self._get_spike_train_features(spike_times=self.spike_times)
            
            except Exception as e:
                logger.error("Run_id=%d: Error in feature extraction: %s", self.run_id, e)
                raise RuntimeError(f"Error in feature extraction for run_id={self.run_id}: {e}") from e

        elif self.data_type == 'spiketrain' and feature_type == 'isi':
            logger.debug("Extracting spike train features using internal methods.")
            self.spike_train_features = self._get_spike_train_features(spike_times=self.spike_times)
            
        logger.info("Feature extraction completed for run_id=%d.", self.run_id)
     
    def _get_spike_train_features(self, spike_times: Union[List[np.ndarray], np.ndarray]) -> pd.DataFrame:
        """
        Compute spike train features from spike times.
        
        Parameters:
        -----------
        spike_times : list of np.ndarray or np.ndarray
            List containing spike times for each neuron or a single array.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing spike train features.
        """
        logger.info("Computing spike train features.")
        if isinstance(spike_times, np.ndarray):
            spike_times = [spike_times]
    
        features_list = []
        # Retrieve duration if it exists; otherwise, compute it
        if not self.config.get('duration', False):
            t_start = self.config.get('t_start')
            t_end = self.config.get('t_end')

            if t_start is None or t_end is None:
                # Check if spike_times is None or empty
                if spike_times is None:
                    raise ValueError("spike_times cannot be None when duration or start/end times are not provided.")
                if not spike_times:
                    raise ValueError("spike_times is empty. Cannot compute duration.")
                    
                # Compute duration from spike_times
                duration = max((st.max() if len(st) > 0 else 0) for st in spike_times)
                self.config['duration'] = duration  # Store as scalar
                logger.debug("Duration or start and end time points are not provided. Using max spike_times (%s) as duration.", duration)
 
            else:
                # Compute duration from t_start and t_end
                if t_start > t_end:
                    raise ValueError(f"Invalid configuration: t_start ({t_start}) is greater than t_end ({t_end}).")
        
                duration = t_end - t_start
                self.config['duration'] = duration  # Store as scalar
                logger.debug("Using provided t_start (%s) and t_end (%s) to compute duration: %s",
                    t_start, t_end, duration)
        else:
            duration = self.config['duration']  # Retrieve as scalar
            logger.debug("Using existing duration from config: %s", duration)

        for idx, st in enumerate(spike_times):
            if len(st) == 0:
                logger.warning("Neuron %d has no spikes.", idx+1)
                features = {
                    'neuron_id': idx+1,
                    'total_spikes': 0,
                    'avg_firing_rate': 0.0,  
                    'mean_isi': None,
                    'median_isi': None,
                    'std_isi': None,
                    'cv_isi': None,
                    'burst_frac': None,
                    'fano_factor': None
                }
            else:

                isi = np.diff(st)
                burst_spikes, iso_spikes, burst_events = separate_spikes(st, burst_th=0.010)
                burst_fraction = (len(burst_spikes)/len(st)) if len(isi) > 0 else None
                mean_isi = np.mean(isi) if len(isi) > 0 else None
                median_isi = np.median(isi) if len(isi) > 0 else None
                std_isi = np.std(isi) if len(isi) > 0 else None
                cv_isi = std_isi / mean_isi if mean_isi != 0 and not np.isnan(mean_isi) else None
                var_isi = np.var(isi) if len(isi) > 0 else None
                fano_factor = var_isi / mean_isi if mean_isi not in [0, None] else None
                features = {
                    'neuron_id': idx+1,
                    'total_spikes': len(st),
                    'avg_firing_rate': len(st) / duration,  
                    'mean_isi': mean_isi,
                    'median_isi': median_isi,
                    'std_isi': std_isi,
                    'cv_isi': cv_isi,
                    'burst_frac': burst_fraction,
                    'fano_factor': fano_factor
                }
            features_list.append(features)
            logger.debug("Neuron %d: %s", idx+1, features)
    
        spike_train_features_df = pd.DataFrame(features_list)
        if spike_train_features_df[['mean_isi', 'median_isi', 'std_isi', 'cv_isi', 'fano_factor']].isnull().all().all():
            logger.warning("All spike train features are None. Returning None.")
            return None

        logger.info("Spike train features computed for %d neurons.", len(features_list))
        return spike_train_features_df

    def save_additional_data(self):
        """
        Save the extracted features to CSV files in the save directory.
        """
        logger.info("Saving additional computed data for run_id=%d.", self.run_id)
        # Save spike features
        if self.spike_features is not None:
            spike_features_path = self.save_dir / f'spike_features_run_{self.run_id}.csv'
            self.spike_features.to_csv(spike_features_path, index=False)
            logger.info("Spike features saved to %s", spike_features_path)
        
        # Save spike train features
        if self.spike_train_features is not None:
            spike_train_features_path = self.save_dir / f'spike_train_features_run_{self.run_id}.csv'
            self.spike_train_features.to_csv(spike_train_features_path, index=False)
            logger.info("Spike train features saved to %s", spike_train_features_path)
