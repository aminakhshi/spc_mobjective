import os
import logging
import warnings
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import neo.io  # Ensure you have neo installed: pip install neo

# Assuming these are available in your environment
from utils.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataParser:
    def __init__(
        self,
        data: Optional[Dict[str, np.ndarray]] = None,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        run_id: int = 0,
        save_dir: str = "results",
        isi_plot_type: str = 'histogram',
        **kwargs
    ):
        """
        Initialize the DataParser with data, parameters, and configurations.

        Parameters:
        -----------
        data : dict, optional
            Dictionary containing 't' (time points) and 'y' (solution array).
        params : dict, optional
            Dictionary of model parameters used in the simulation.
        config : dict, optional
            Dictionary of simulation configurations.
        run_id : int
            Unique identifier for the simulation run.
        save_dir : str, optional
            Directory where results will be saved. Defaults to "results".
        isi_plot_type : str, optional
            Type of ISI plot: 'histogram' or 'kde'. Defaults to 'histogram'.
        **kwargs : dict
            Additional keyword arguments:
                - cutoff_fr (float): Fraction of data to remove as transient. Defaults to 0.25.
                - save_result (bool): Whether to save the results. Defaults to True.
                - threshold (float): Threshold for spike detection. Defaults to 0.0.
                - distance (int): Minimum distance between peaks in samples. Defaults to 20.
        """
        self.t = data.get('t', np.array([])) if data else np.array([])
        self.y = data.get('y', np.array([])) if data else np.array([])
        self.params = params if params else {}
        self.config = config if config else {}
        self.run_id = run_id
        self.save_dir = save_dir
        self.isi_plot_type = isi_plot_type

        # Additional parameters with defaults
        self.cutoff_fr = kwargs.get('cutoff_fr', 0.25)
        self.save_result = kwargs.get('save_result', True)
        self.threshold = kwargs.get('threshold', 0.0)
        self.distance = kwargs.get('distance', 20)

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Placeholder for extracted features
        self.spike_features = None
        self.spike_train_features = None
        self.sta_segments = None

    def prepare_data(
        self,
        fname: str,
        ftype: str = 'spiketimes',
        parsed_segment: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Read and parse recording data from a file.

        Parameters:
        -----------
        fname : str
            File path to the recording data.
        ftype : str, optional
            File type of the recording. Accepts ['spiketimes', 'membrane_potential', 'ap'].
            Defaults to 'spiketimes'.
        parsed_segment : bool, optional
            If True, returns parsed data suitable for further processing.
            If False, returns raw data segments. Defaults to False.
        additional_params : dict, optional
            Additional information with the data such as sampling rate, stimulus, etc.

        Returns:
        --------
        dict
            Parsed data containing recordings, stimuli, metadata, etc.

        Notes:
        ------
            For 'ap' type, the function only supports recordings from .smr files (Spike2).
        """
        logger.info(f"Preparing data from file: {full_path} with type: {ftype}")

        if ftype == 'ap' and fname.endswith('.smr'):
            logger.debug("Initializing reading membrane potential data from Spike2 file.")
            return self._read_spike2(fname, parsed_segment)
        elif ftype in ['spiketimes', 'membrane_potential']:
            # Implement other file types as needed
            logger.error(f"File type '{ftype}' is not supported yet.")
            raise NotImplementedError(f"File type '{ftype}' is not supported.")
        else:
            logger.error(f"Unsupported file type: {ftype}")
            raise ValueError(f"Unsupported file type: {ftype}")

    def _read_spike2(self, fname: str, parsed_segment: bool) -> Dict[str, Any]:
        """
        Private method to read Spike2 .smr files.

        Parameters:
        -----------
        fname : str
            Full file path to the .smr file.
        parsed_segment : bool
            Determines the format of the returned data.

        Returns:
        --------
        dict
            Parsed data containing recordings, stimuli, metadata, etc., if parsed_segment is True.
            Otherwise, returns raw data segments.
        """
        import warnings
        import neo.io
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                reader = neo.io.Spike2IO(filename=fname)
                logger.info(f"Successfully initialized Spike2IO for file: {fname}")
            except Exception as e:
                logger.warning(f"Failed to initialize Spike2IO with default parameters: {e}. Trying alternative settings.")
                reader = neo.io.Spike2IO(filename=fname, try_signal_grouping=False)

            block = reader.read_block(lazy=False)
        
        data_segments = []

        for seg_idx, segment in enumerate(block.segments):
            logger.debug(f"Processing segment {seg_idx}...")
            segments = {
                "segment_index": seg_idx,
                "analogsignals": [],
                "events": [],
                "epochs": [],
                "spiketrains": []
            }

            # Process Analog Signals
            logger.info(f"Processing analog signals in segment {seg_idx}...")
            for idx, signal in enumerate(segment.analogsignals):
                signals = {
                    "index": idx,
                    "signal_type": 'analog',
                    "sampling_rate": float(signal.sampling_rate.rescale('Hz')),
                    "duration": float((signal.t_stop - signal.t_start).rescale('s')),
                    "units": str(signal.units),
                    "shape": signal.shape,
                    'time': np.array(signal.times.rescale('s').magnitude),
                    'data': np.squeeze(signal.magnitude)
                }

                logger.debug(f"Analog Signal [{idx}]: {signals['units']}, Shape: {signals['shape']}")
                segments["analogsignals"].append(signals)

            # Process Events
            logger.info(f"Processing events in segment {seg_idx}...")
            for idx, event in enumerate(segment.events):
                events = {
                    "index": idx,
                    "units": str(event.units),
                    "times": np.array(event.times.rescale('s').magnitude),
                    "labels": event.labels if hasattr(event, "labels") else None,
                }
                logger.debug(f"Event [{idx}]: Units: {events['units']}, Number of Events: {len(events['times'])}")
                segments["events"].append(events)

            # Process Epochs
            logger.info(f"Processing epochs in segment {seg_idx}...")
            for idx, epoch in enumerate(segment.epochs):
                epochs = {
                    "index": idx,
                    "times": epoch.times.rescale('s').magnitude,
                    "durations": epoch.durations.rescale('s').magnitude,
                    "labels": epoch.labels if hasattr(epoch, "labels") else None,
                }
                logger.debug(f"Epoch [{idx}]: Number of Epochs: {len(epochs['times'])}")
                segments["epochs"].append(epochs)

            # Process Spike Trains
            logger.info(f"Processing spike trains in segment {seg_idx}...")
            for idx, spiketrain in enumerate(segment.spiketrains):
                spiketrains = {
                    "index": idx,
                    "times": spiketrain.times.rescale('s').magnitude,
                    "units": str(spiketrain.units),
                }
                logger.debug(f"Spike Train [{idx}]: Units: {spiketrains['units']}, Number of Spikes: {len(spiketrains['times'])}")
                segments["spiketrains"].append(spiketrains)

            data_segments.append(segments)

        if parsed_segment:
            logger.info("Parsing segments into structured data...")
            parsed_data = {
                "recordings": [],
                "stimuli": [],
                "metadata": {}
            }

            for segment in data_segments:
                for signal in segment.get("analogsignals", []):
                    if "nA" in signal["units"]:
                        # Current recording
                        parsed_data["stimuli"].append({
                            "data": signal["data"],
                            'time': signal['time'],
                            "sampling_rate": signal["sampling_rate"],
                            "duration": signal["duration"],
                            "shape": signal["shape"]
                        })
                    elif "V" in signal["units"] and signal["data"].ndim == 1:
                        # Voltage recording
                        parsed_data["recordings"].append({
                            "data": signal["data"],
                            'time': signal['time'],
                            "sampling_rate": signal["sampling_rate"],
                            "duration": signal["duration"],
                            "channels": signal["shape"][1] if len(signal["shape"]) > 1 else 1
                        })
                    
                # # Process Events
                # for event in segment.get("Events", []):
                    # parsed_data.setdefault("events", []).append({
                        # "name": event["name"],
                        # "times": event["times"].magnitude if hasattr(event["times"], "magnitude") else [],
                        # "labels": event["labels"]
                    # })
                
                # # Process Spiketrains (if any)
                # for spiketrain in summary.get("Spiketrains", []):
                    # parsed_data.setdefault("spiketrains", []).append({
                        # "name": spiketrain["name"],
                        # "times": spiketrain["times"]
                    # })
                    
            logger.info("Data parsing complete.")
            return parsed_data
        else:
            logger.info("Returning raw data segments.")
            return {"segments": data_segments}
# =============================================================================
# 
# =============================================================================
    def extract_features(self, fs: float = 10000, sta_win: float = 10.0, spkt_ref: str = 'peak_t'):
        """
        Perform spike feature extraction and spike train feature extraction.

        Parameters:
        -----------
        fs : float, optional
            Sampling frequency in Hz. Defaults to 10000.
        sta_win : float, optional
            Spike-triggered average window in milliseconds. Defaults to 10.0.
        spkt_ref : str, optional
            Reference for spike times. Options: ['peak_t', 'threshold_t', 'trough_t', 'adp_t'].
            Defaults to 'peak_t'.
        """
        logger.info("Starting feature extraction...")
        self.spike_features, self.spike_train_features, self.sta_segments = self._sta(
            time=self.t,
            rec=self.y,
            fs=fs,
            sta_win=sta_win,
            spkt_ref=spkt_ref
        )
        logger.info("Feature extraction completed.")

    def _sta(
        self,
        time: np.ndarray,
        rec: np.ndarray,
        stim: Optional[np.ndarray] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        fs: float = 10000,
        sta_win: float = 10.0,
        spkt_ref: str = 'peak_t'
    ) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
        """
        Private method to compute Spike-Triggered Average (STA) and extract features.

        Parameters:
        -----------
        time : np.ndarray
            Time points.
        rec : np.ndarray
            Recording data (e.g., membrane potential).
        stim : np.ndarray, optional
            Stimulus data. Defaults to zeros if not provided.
        start : float, optional
            Start time for analysis. Defaults to the first time point.
        end : float, optional
            End time for analysis. Defaults to the last time point.
        fs : float, optional
            Sampling frequency in Hz. Defaults to 10000.
        sta_win : float, optional
            STA window size in milliseconds. Defaults to 10.0.
        spkt_ref : str, optional
            Reference for spike times. Defaults to 'peak_t'.

        Returns:
        --------
        tuple
            (spikes_df, spike_train_df, sta_segments)
        """
        logger.debug("Initializing Spike Feature Extractor and Spike Train Feature Extractor.")
        start = start if start is not None else time[0]
        end = end if end is not None else time[-1]

        if stim is None:
            stim = np.zeros_like(rec)

        sfx = SpikeFeatureExtractor(start=start, end=end)
        spikes_df = sfx.process(t=time, v=rec, i=stim)
        stfx = SpikeTrainFeatureExtractor(start=start, end=end)
        spike_train_df = stfx.process(t=time, v=rec, i=stim, spikes_df=spikes_df)

        logger.debug("Spike and Spike Train features extracted.")

        # Spike-triggered average estimation
        win_size = int(sta_win * (fs / 1000))  # Convert ms to samples
        logger.debug(f"STA window size in samples: {win_size}")

        # Extract spike times based on reference
        if spkt_ref.endswith('t'):
            spike_times = spikes_df[spkt_ref].values
            spike_indices = np.searchsorted(time, spike_times)
        elif spkt_ref.endswith('index'):
            spike_indices = spikes_df[spkt_ref].values
        else:
            logger.error(f"Invalid spike reference: {spkt_ref}")
            raise ValueError(f"Invalid spike reference: {spkt_ref}")

        logger.debug(f"Number of spikes detected: {len(spike_indices)}")

        # Initialize array to store STA segments
        sta_segments = np.zeros((len(spike_indices), 2 * win_size + 1))

        logger.debug("Extracting STA segments.")
        for i, spike_idx in enumerate(spike_indices):
            if spike_idx - win_size >= 0 and spike_idx + win_size < len(rec):
                sta_segments[i] = rec[spike_idx - win_size: spike_idx + win_size + 1]

        logger.debug("STA segments extraction complete.")
        return spikes_df, spike_train_df, sta_segments

    def save_results(self):
        """
        Save the extracted features and STA segments to the save directory.
        """
        if not self.save_result:
            logger.info("Saving of results is disabled.")
            return

        logger.info("Saving results...")

        # Save spike features
        if self.spike_features is not None:
            spike_features_path = os.path.join(self.save_dir, f'spike_features_run_{self.run_id}.csv')
            self.spike_features.to_csv(spike_features_path, index=False)
            logger.debug(f"Spike features saved to {spike_features_path}")

        # Save spike train features
        if self.spike_train_features is not None:
            spike_train_features_path = os.path.join(self.save_dir, f'spike_train_features_run_{self.run_id}.csv')
            self.spike_train_features.to_csv(spike_train_features_path, index=False)
            logger.debug(f"Spike train features saved to {spike_train_features_path}")

        # Save STA segments
        if self.sta_segments is not None:
            sta_path = os.path.join(self.save_dir, f'sta_segments_run_{self.run_id}.npy')
            np.save(sta_path, self.sta_segments)
            logger.debug(f"STA segments saved to {sta_path}")

        logger.info("All results have been saved successfully.")

    def run_parallel_feature_extraction(
        self,
        segments: List[Dict[str, Any]],
        fs: float = 10000,
        sta_win: float = 10.0,
        spkt_ref: str = 'peak_t',
        max_workers: int = 4
    ):
        """
        Perform feature extraction in parallel across multiple data segments.

        Parameters:
        -----------
        segments : list of dict
            List of data segments to process.
        fs : float, optional
            Sampling frequency in Hz. Defaults to 10000.
        sta_win : float, optional
            STA window size in milliseconds. Defaults to 10.0.
        spkt_ref : str, optional
            Reference for spike times. Defaults to 'peak_t'.
        max_workers : int, optional
            Maximum number of parallel workers. Defaults to 4.
        """
        logger.info("Starting parallel feature extraction...")
        results = []

        def process_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
            logger.debug(f"Processing segment {segment['segment_index']} in parallel.")
            # Assuming each segment has 'analogsignals' with voltage recordings
            recordings = [s for s in segment.get("analogsignals", []) if "V" in s["units"]]
            if not recordings:
                logger.warning(f"No voltage recordings found in segment {segment['segment_index']}. Skipping.")
                return {}

            # For simplicity, process the first voltage recording
            recording = recordings[0]
            time = recording['time']
            rec = recording['data']
            spikes_df, spike_train_df, sta_segments = self._sta(
                time=time,
                rec=rec,
                fs=fs,
                sta_win=sta_win,
                spkt_ref=spkt_ref
            )
            return {
                "spikes_df": spikes_df,
                "spike_train_df": spike_train_df,
                "sta_segments": sta_segments
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {executor.submit(process_segment, seg): seg for seg in segments}
            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as exc:
                    logger.error(f"Segment {segment['segment_index']} generated an exception: {exc}")

        # Aggregate results
        if results:
            self.spike_features = pd.concat([res['spikes_df'] for res in results], ignore_index=True)
            self.spike_train_features = pd.concat([res['spike_train_df'] for res in results], ignore_index=True)
            self.sta_segments = np.vstack([res['sta_segments'] for res in results])
            logger.info("Parallel feature extraction completed and results aggregated.")
        else:
            logger.warning("No results obtained from parallel feature extraction.")

    def plot_isi(self):
        """
        Plot Inter-Spike Interval (ISI) histogram or KDE based on configuration.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.spike_train_features is None:
            logger.error("Spike train features have not been extracted. Cannot plot ISI.")
            return

        isi = self.spike_train_features['isi']

        plt.figure(figsize=(10, 6))
        if self.isi_plot_type == 'histogram':
            plt.hist(isi, bins=50, color='blue', alpha=0.7)
            plt.title('Inter-Spike Interval (ISI) Histogram')
            plt.xlabel('ISI (s)')
            plt.ylabel('Frequency')
        elif self.isi_plot_type == 'kde':
            sns.kdeplot(isi, shade=True, color='blue')
            plt.title('Inter-Spike Interval (ISI) KDE')
            plt.xlabel('ISI (s)')
            plt.ylabel('Density')
        else:
            logger.error(f"Unsupported ISI plot type: {self.isi_plot_type}")
            return

        plt.grid(True)
        plt.tight_layout()
        isi_plot_path = os.path.join(self.save_dir, f'isi_plot_run_{self.run_id}.png')
        plt.savefig(isi_plot_path)
        plt.close()
        logger.info(f"ISI plot saved to {isi_plot_path}")


# Example usage:
if __name__ == "__main__":
    # Initialize RecParser with optional data, params, config
    parser = RecParser(
        run_id=1,
        save_dir="results",
        isi_plot_type='histogram',
        threshold=0.5,
        distance=30
    )

    # Prepare data
    parsed_data = parser.prepare_data(
        fname='04_data_a.smr',
        ftype='ap',
        parsed_segment=True,
        additional_params={
            'data_path': '/home/amin/khadralab/allensdk/data',
            'exp_date': '10152013'
        }
    )

    # Extract features
    parser.extract_features(fs=10000, sta_win=10.0, spkt_ref='peak_t')

    # Save results
    parser.save_results()

    # Plot ISI
    parser.plot_isi()

    # Alternatively, for parallel feature extraction:
    # segments = parsed_data.get("segments", [])
    # parser.run_parallel_feature_extraction(segments, max_workers=8)
    # parser.save_results()
    # parser.plot_isi()
