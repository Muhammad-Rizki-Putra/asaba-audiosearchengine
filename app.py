import os
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create the 'uploads' folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Core Audio Fingerprinting Logic (Your Function) ---
def fingerprint_song(file_path):
    """
    Generates a landmark-based fingerprint for a single audio file.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        dict: A dictionary of {hash: timestamp}
    """
    try:
        y, sr = librosa.load(file_path)

        # 1. Create Spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 2. Find Peaks
        neighborhood_size = 15
        local_max = maximum_filter(S_db, footprint=np.ones((neighborhood_size, neighborhood_size)), mode='constant')
        detected_peaks = (S_db == local_max)
        amplitude_threshold = -50.0
        peaks = np.where((detected_peaks) & (S_db > amplitude_threshold))
        
        if not peaks[0].any():
            return {}

        # 3. Structure Peaks
        n_fft = (D.shape[0] - 1) * 2
        peak_freqs_at_peaks = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peaks[0]]
        peak_times = librosa.frames_to_time(frames=peaks[1], sr=sr, n_fft=n_fft)
        peaks_list = list(zip(peak_times, peak_freqs_at_peaks))
        sorted_peaks = sorted(peaks_list, key=lambda p: p[0])

        # 4. Create Hashes
        song_fingerprint = {}
        TARGET_ZONE_START_TIME = 0.1
        TARGET_ZONE_TIME_DURATION = 0.8
        TARGET_ZONE_FREQ_WIDTH = 200

        for i, anchor_peak in enumerate(sorted_peaks):
            anchor_time, anchor_freq = anchor_peak
            t_min = anchor_time + TARGET_ZONE_START_TIME
            t_max = t_min + TARGET_ZONE_TIME_DURATION
            f_min = anchor_freq - TARGET_ZONE_FREQ_WIDTH
            f_max = anchor_freq + TARGET_ZONE_FREQ_WIDTH
            
            for j in range(i + 1, len(sorted_peaks)):
                target_peak = sorted_peaks[j]
                target_time, target_freq = target_peak
                if target_time > t_max:
                    break
                if t_min <= target_time <= t_max and f_min <= target_freq <= f_max:
                    time_delta = target_time - anchor_time
                    h = hash((anchor_freq, target_freq, time_delta))
                    song_fingerprint[h] = anchor_time
                    
        return song_fingerprint

    except Exception as e:
        print(f"Could not process {file_path}. Error: {e}")
        return {}

# --- API Endpoint ---
@app.route('/fingerprint', methods=['POST'])
def generate_fingerprint_endpoint():
    """API endpoint to receive a song and return its fingerprint."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        fingerprint_data = fingerprint_song(file_path)
        
        os.remove(file_path)

        if not fingerprint_data:
            return jsonify({
                "message": "Could not generate fingerprint. The audio might be silent or too short.",
                "fingerprint": {}
            }), 200

        return jsonify(fingerprint_data)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
