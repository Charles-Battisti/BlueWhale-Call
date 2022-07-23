# BlueWhale-Call
Automatically detect Blue Whale calls from hydrophone data. Built to work on an in situ instrument with a microcontroller processor. This method attempts to isolate Blue Whale calls by identifying the ridge of the call.

Method:
1. Create spectrogram using numpy fast fourier transform. Restrict the output to the frequency band of interest.
2. To emphasize the calls, a 45 degree kernel is passed over the spectrogram.
3. Remove most remaining noise using a high pass filter based on the 95% quantile.
4. Identify groups using breadth first search.
5. Build ridge of a group by computing the median of each timestamp (column), then smoothing the medians using a moving average.
6. Identify candidate whale calls using the duration of the groups and the change in frequency between the first and second half of the group.
