from collections import deque

import pyaudio
import numpy as np
from scipy import signal


class AudioStream:
    """Input audio stream."""
    def __init__(self, sample_rate=44100, chunk_size=1024, fft_size=1024, format=pyaudio.paFloat32, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.buffer = deque(np.zeros(fft_size, dtype=np.float32), maxlen=fft_size)

    def start(self):
        try:
            host_params = self.pa.get_default_input_device_info()
        except IOError as e:
            print(f'{e}')
        
        print(f'Default input device parameters: {host_params}')

        self.stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
    
    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
    
    def update(self):
        try:
            audio_data = np.frombuffer(
                self.stream.read(self.chunk_size, exception_on_overflow=False), 
                dtype=np.float32
            )
            self.buffer.extend(audio_data)
        except Exception as e:
            print(f"Audio error: {e}")
            return 0.5
        

class AudioAnalyzer:
    """Logarithmic audio spectrum analyzer."""
    def __init__(self, n_bins=256, min_freq=20, max_freq=20000, sample_rate=44100, fft_size=1024, window='hann', min_db=-60, max_db=30):
        self.n_bins = n_bins
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = signal.get_window(window, fft_size)
        self.min_db = min_db
        self.max_db = max_db

        self.fft_freqs = np.fft.rfftfreq(fft_size, d=1/sample_rate)
        self.log_freqs = self.create_log_frequency_bins(
            n_bins // 2, min_freq, max_freq, sample_rate, fft_size
        )
        
        # Initialize buffer for storing frequency data
        self.spectrum = np.zeros(n_bins)
        
        # Animation parameters
        self.alpha = 0.5
        self.smoothed_spectrum = np.zeros(n_bins)

        self.min_rms_threshold = 0.6

        self.beat_band = self.get_band_indices(200, 1000)
        self.beat_history = deque(maxlen=60)
        self.beat_cooldown_frames = 20
        self.beat_cooldown_counter = 0
        self.beat_detected = False

        self.chrom_aberration_strength = 0.0


    def create_log_frequency_bins(self, n_bins, min_freq=20, max_freq=20000, sample_rate=44100, fft_size=1024):
        """
        Create logarithmically spaced frequency bins suitable for audio analysis.

        Args:
            n_bins (int) : Number of frequency bins to create
            min_freq (float) : Minimum frequency in Hz, default 20 Hz
            max_freq (float) : Maximum frequency in Hz, default 20 kHz
            sample_rate (int) : Audio sample rate in Hz, default 44.1 kHz

        Returns:
            - center_frequencies: Array of bin center frequencies
        """
        # Compute number of octaves and bins per octave
        n_octaves = np.log2(max_freq / min_freq)
        bins_per_octave = n_bins / n_octaves

        # Create logarithmically spaced frequencies
        n_edges = int(np.ceil(bins_per_octave * n_octaves)) + 1
        log_bin_edges = min_freq * (2 ** (np.arange(n_edges) / bins_per_octave))
        center_frequencies = np.sqrt(log_bin_edges[:-1] * log_bin_edges[1:])

        return center_frequencies
    
    def get_band_indices(self, low, high):
        return np.where((self.log_freqs >= low) & (self.log_freqs <= high))[0]
    
    def compute_band_energy(self, spectrum, indices):
        return np.mean(spectrum[indices]) if len(indices) > 0 else 0
    
    def detect_beat(self, energy, history, threshold):
        if len(history) < history.maxlen:
            history.append(energy)
            return False
        avg_energy = np.mean(history)
        history.append(energy)
        return energy > threshold * avg_energy
    
    def update(self, audio_data):
        """
        Analyze a single frame of audio and updates the spectrum buffer.
        
        Args:
            audio_data (np.ndarray) : Array of audio samples
        
        """
        # Zero-padding
        if len(audio_data) < self.fft_size:
            audio_data = np.pad(
                audio_data, 
                (0, self.fft_size - len(audio_data)),
                mode='constant',
                constant_values=0
        )
        
        # Apply window function and compute FFT
        windowed = audio_data * self.window
        fft_magnitude = np.abs(np.fft.rfft(windowed, n=self.fft_size))
        fft_power = fft_magnitude ** 2

        # Interpolate to log frequency scale
        log_power = np.interp(self.log_freqs, self.fft_freqs, fft_power)

        # Convert magnitude to dB scale
        log_db = 10 * np.log10(log_power + 1e-10)
        log_db = np.clip(log_db, self.min_db, self.max_db)

        # Reflect across y-axis for symmetry
        log_db = np.concatenate([log_db[::-1], log_db])

        # Normalize and update spectrum buffer
        self.spectrum = (log_db - self.min_db) / (self.max_db - self.min_db)
        self.spectrum = np.clip(self.spectrum, 0, 1)
        self.smoothed_spectrum = (
            self.alpha * self.smoothed_spectrum 
            + (1 - self.alpha) * self.spectrum
        )
        
        beat_detected = False
        overall_rms = np.sqrt(np.mean(self.spectrum ** 2))
        if overall_rms > self.min_rms_threshold:
            print(f'overall_rms: {overall_rms:.4f}')
            beat_energy = self.compute_band_energy(self.spectrum, self.beat_band)
            energy_history = np.mean(self.beat_history)
            beat_threshold = energy_history * 1.8
            beat_detected = self.detect_beat(beat_energy, self.beat_history, beat_threshold)

        if self.beat_cooldown_counter > 0:
            self.beat_cooldown_counter -= 1

        self.beat_detected = False

        if self.beat_cooldown_counter == 0 and beat_detected:
            self.beat_detected = True
            self.beat_cooldown_counter = self.beat_cooldown_frames
            self.chrom_aberration_strength = 1.0
        
        self.chrom_aberration_strength *= 0.8


if __name__ == '__main__':
    print(f'{np.__version__=}')
