import numpy as np
import librosa
from scipy.fftpack import dct

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# 34 features:
# 12 MFCC means
# 12 MFCC stds
# Spectral centroid mean/std
# Spectral bandwidth mean/std
# Spectral rolloff mean/std
# Zero-crossing rate mean/std
# RMS energy mean/std

features = np.zeros((10, 100, 34))

for i, genre in enumerate(genres):
    for j in range(100):

        audio_path = f'C:/Users/User/Desktop/MLProject/genres/{genre}/{genre}.{j:05d}.wav'

        # ----------------------------
        # Load audio
        # ----------------------------
        y, sr = librosa.load(audio_path)

        # ----------------------------
        # MFCC extraction (manual)
        # ----------------------------
        pre_emphasis = 0.97
        y_preemphasized = np.append(
            y[0],
            y[1:] - pre_emphasis * y[:-1]
        )

        frame_size = 0.025
        frame_stride = 0.01

        frame_length = int(round(frame_size * sr))
        frame_step = int(round(frame_stride * sr))

        signal_length = len(y_preemphasized)

        num_frames = int(
            np.ceil(
                np.abs(signal_length - frame_length) / frame_step
            )
        )

        pad_signal_length = num_frames * frame_step + frame_length
        pad_signal = np.append(
            y_preemphasized,
            np.zeros(pad_signal_length - signal_length)
        )

        indices = (
            np.tile(np.arange(frame_length), (num_frames, 1))
            + np.tile(
                np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)
            ).T
        )

        frames = pad_signal[indices.astype(np.int32)]
        frames *= np.hamming(frame_length)

        NFFT = 512

        mag_frames = np.abs(np.fft.rfft(frames, NFFT))
        pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

        nfilt = 40

        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)

        mel_points = np.linspace(low_freq_mel,
                                 high_freq_mel,
                                 nfilt + 2)

        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((nfilt, NFFT // 2 + 1))

        for m in range(1, nfilt + 1):

            left = bins[m - 1]
            center = bins[m]
            right = bins[m + 1]

            for k in range(left, center):
                fbank[m - 1, k] = (k - left) / (center - left)

            for k in range(center, right):
                fbank[m - 1, k] = (right - k) / (right - center)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.maximum(filter_banks, np.finfo(float).eps)
        filter_banks = 20 * np.log10(filter_banks)

        num_ceps = 12

        mfcc = dct(
            filter_banks,
            type=2,
            axis=1,
            norm='ortho'
        )[:, :num_ceps]

        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)

        # ----------------------------
        # Additional spectral features
        # ----------------------------

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )

        zcr = librosa.feature.zero_crossing_rate(y)

        rms = librosa.feature.rms(y=y)

        spectral_features = np.array([
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ])

        features[i, j] = np.concatenate((
            mfcc_mean,
            mfcc_std,
            spectral_features
        ))

# Save
np.save("features_34.npy", features)