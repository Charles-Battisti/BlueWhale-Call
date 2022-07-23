# internal python libraries
from copy import copy

# 3rd party libraries
import numpy as np

# internally developed methods
import grouper_bfs as gp


large_kernel_45 = np.array([[0, 1, 2, 3, 4, 5, 6],
                            [-1, 0, 1, 2, 3, 4, 5],
                            [-2, -1, 0, 1, 2, 3, 4],
                            [-3, -2, -1, 0, 1, 2, 3],
                            [-4, -3, -2, -1, 0, 1, 2],
                            [-5, -4, -3, -2, -1, 0, 1],
                            [-6, -5, -4, -3, -2, -1, 0]])

blue_whale_call_statistics = {'min_freq': 30,
                              'max_freq': 50,
                              'time diff': 500,
                              'frequency diff': 0.5}


# function from: https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, min_freq=None, max_freq=None, eps=1e-14):
    """
    Build a spectrogram from audio data using fast fourier transform (fft).
    Spectrogram will have sample_rate * window_ms
    
    :param samples: (1-D array-like) audio data
    :param sample_rate: (int) sampling frequency in herz (hz)
    :param stride_ms: (float) offset from one window to the next window in herz.
                      Should be less than window_ms so that the windows overlap.
    :param window_ms: (float) window size in ms for the fourier transform
    :param min_freq: (float) minimum frequency of interest in the spectrogram
    :param max_freq: (float) maximum frequency of interest in spectrogram
    :param eps: (float) small value to ensure that log(0) does not occur.
    
    :return: list of frequencies associated with the y axis of the spectrogram and the spectrogram.
    """
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    n = 20000

    fft = np.fft.rfft(windows * weighting, n=n, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2

    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = (20000 / n if n else 1) * float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind_low = np.where(min_freq <= freqs)[0][0]
    ind_high = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[ind_low:ind_high, :] + eps)
    return freqs[ind_low:ind_high], specgram


def sub_matrix_generator(matrix, sub_matrix_shape):
    """
    Yields a sub-matrix of shape sub_matrix_shape taken from the matrix.
    The position of the sub-matrix will incrementally move vertically first, then horizontally.
    Used to convolve a kernel over a matrix.
    
    :param matrix: (2-D ndarray) matrix from which sub-matricies are drawn.
    :param sub_matrix_shape: (tuple) the shape of the sub-matrix.
    
    :yield: x and y positions of the sub-matrix in the new convolved matrix and the sub-matrix itself.
    """
    
    for x in range(matrix.shape[0] - (sub_matrix_shape[0] - 1)):
        for y in range(matrix.shape[1] - (sub_matrix_shape[1] - 1)):
            yield x, y, matrix[x:x + sub_matrix_shape[0], y:y + sub_matrix_shape[0]]


def convolve(matrix, kernel):
    """
    Convolves a kernel over a matrix.
    
    :param matrix: (2-D ndarray) matrix over which the kernel is run.
    :param kernel: (2-D ndarray) smaller matrix which is convolved over matrix.
    
    :return: (2-D ndarray) convolved matrix. This matrix will be (kernel.shape[0] -1, kernel.shape[1] - 1) smaller than matrix.
    """
    
    shape_adjustment = int(np.ceil(kernel.shape[0] / 2))
    index_adjustment = int(np.floor(kernel.shape[0] / 2))
    kernelized_matrix = np.zeros((matrix.shape[0] - (kernel.shape[0] - 1), matrix.shape[1] - (kernel.shape[1] - 1)))
    generator = sub_matrix_generator(matrix, kernel.shape)
    for x_pos, y_pos, sub_matrix in generator:
        kernelized_matrix[x_pos, y_pos] = (sub_matrix * kernel).sum()
    return kernelized_matrix


def detect_whale_call_from_audio(audio_data, audio_frequency, whale_statistics_dict=blue_whale_call_statistics,
                                 conv_kernel=large_kernel_45, min_group_size=400, quantile_threshold=0.95):
    """
    Detects Blue Whale calls in audio (hydrophone) data.
    Process:
    1. Use fourier transform to conver audio data to frequency domain. Select frequency range of relevant whale calls.
    2. Pass a convolutional kernel over the frequency data to emphasize whale calls and reduce irrelevant signals.
    3. Use a high pass filter (95% quantile) to remove irrelevant signals.
    4. Group remaining data.
    5. Develop "ridge" (moving average) for each group.
    6. Apply simple test statistics to isolate whale calls.
    
    :param audio_data: (1-D array-like) audio data.
    :param audio_frequency: (int) number of herz (hz) that audio data was collected.
    :param whale_statistics_dict: (dict) specifies the frequency range of calls for fourier transform,
                                  the minimum length of a group to be considered a call, and the mininum
                                  frequency change of a group to be considered a whale call.
    :param conv_kernel: (2-D ndarray) convolutional kernel used to emphasize whale calls.
    :param min_group_size: (int) minimum members of a group necessary to be considered a viable call candidate.
                           This parameter is used to skip groups that are too small to be considered a whale call
                           before other statistics are comupted and tested.
    :param quantile_threshold: (float between 0 and 1) threshold to create high pass filter to remove irrelevant noise
                               in convolved matrix.
    
    :return: (dict) group objects (under 'group' key) and associated statistics (under 'statistics' key) which were
             identified to be whale calls.
    """
    
    freq, spec = spectrogram(audio_data, audio_frequency, window_ms=4000.0,
                             min_freq=whale_statistics_dict['min_freq'], max_freq=whale_statistics_dict['max_freq'])

    # kernelize spectrogram
    kernelized_spec = convolve(spec, conv_kernel)

    # scale the output
    kernelized_spec = (kernelized_spec - kernelized_spec.min()) / (kernelized_spec.max() - kernelized_spec.min())

    # filter results
    filtered_kspec = copy(kernelized_spec)
    quantile_band = np.quantile(kernelized_spec, quantile_threshold)  # 0.95
    filtered_kspec[filtered_kspec < quantile_band] = 0
    filtered_kspec[filtered_kspec > quantile_band] = 1

    # build groups
    groups = gp.group_array(filtered_kspec, threshold=0.95)

    # filter out small groups
    large_groups = [g for g in groups if len(g.group_members()) > min_group_size]
    x = np.arange(len(large_groups))
    group_dict = {num + 1: group for group, num in zip(large_groups, x)}

    # build walking means of each group
    walking_medians = {key: value.walking_median() for key, value in group_dict.items()}

    # smooth the walking means
    smooth_medians = {key: gp.moving_average(np.array(value), 10) for key, value in walking_medians.items()}

    # compute statistics
    statistics_dict = {}
    for key, value in smooth_medians.items():
        if len(value) > 0:
            statistics_dict[key] = gp.group_statistics(value)

    identified_calls = {'group': [], 'statistics': []}
    for key, value in statistics_dict.items():
        if (statistics_dict[key]['time diff'] > whale_statistics_dict['time diff'] and
                statistics_dict[key]['frequency diff'] > whale_statistics_dict['frequency diff']):
            identified_calls['group'].append(group_dict[key])
            identified_calls['statistics'].append(value)
    return identified_calls
