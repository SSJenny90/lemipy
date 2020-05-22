from scipy import signal
from itertools import repeat

def remove_signal(bandwidth, multiple=50, range_start=None, range_end=None, sample_rate=1000):
    nyq = sample_rate * 0.5
    if not range_end:
        range_end = nyq
    return [(x,y) for x,y in zip(range(int(range_start),int(range_end),int(multiple)),repeat(bandwidth))]

def remove_50hz_odd(bandwidth,sample_rate=1000):
    return remove_signal(bandwidth, 100, 50, sample_rate=sample_rate)

def remove_50hz_even(bandwidth,sample_rate=1000):
    return remove_signal(bandwidth, 100, 100, sample_rate=sample_rate)

def hp(cutoff):
    return [(0,cutoff)]


def comb(filter_specs, sample_rate):
    """Create a comb filter in a similar manner to Matlab's IdealFilter

    Parameters
    ----------

    filter_specs : list of tuples
        List of tuples where item[0] is the target freq and item[1] is the desired bandwidth to filter

    returns the numerator (b) and denominator (a) polynomials of the comb filter
    """
    b, a = None, None #in case all specified filters are above nyquist freq
    
    for i, specs in enumerate(sorted(filter_specs)):
        freq = specs[0]
        bw = specs[1]

        if freq > sample_rate / 2:
            # make sure only filters below the nyquist freq are applied
            # otherwise will throw error
            continue

        if i == 0:
            b, a = apply_filter(freq, bw, sample_rate)
        else:
            tmpb, tmpa = apply_filter(freq, bw, sample_rate)
            a = signal.convolve(a,tmpa)
            b = signal.convolve(b,tmpb)
    return b, a

def apply_filter(freq, bandwidth, sample_rate):
    if freq == 0:  # this is a highpass
        b, a = high_pass(cutoff_freq=bandwidth,sample_rate=sample_rate)
    elif freq == sample_rate/2: # this is a low pass
        b, a = low_pass(cutoff_freq=freq - bandwidth/2, sample_rate=sample_rate)
    else:   
        b, a = notch(freq,bandwidth,sample_rate)
    
    return b, a



def notch(freq,bandwidth,sample_rate=1000):
    """Creates a -3dB notch filter centered at "freq" with a width of "bandwidth"
    Returns the filtered data.	
    """
    return signal.iirnotch(freq,freq/bandwidth, fs=sample_rate)

def bandstop(low_cut,high_cut,sample_rate=1000,order=3):
    return signal.butter(order, [low_cut,high_cut], 'bandstop',fs=sample_rate)
    # b, a = butter(order, [low, high], btype='bandstop')
    # return signal.iirnotch(freq,freq/bandwidth, fs=sample_rate)

def high_pass(cutoff_freq,order=2,sample_rate=1000):
    return signal.butter(order, cutoff_freq, btype='hp', fs=sample_rate)

def low_pass(cutoff_freq,order=2,sample_rate=1000):
    return signal.butter(order, cutoff_freq, fs=sample_rate)

def auto_filter(self,bandwidth=4,plot_peaks=False):
    """Experimental"""
    f, Pxx_den = signal.welch(self.data['Ex'],fs=self.sample_rate,nperseg=self.sample_rate)
    peaks, properties = signal.find_peaks(Pxx_den,threshold=1,distance=20)
    
    fig,ts = plt.subplots(1)
    ts.plot(self.data['Ex'],'b')

    for freq in f[peaks]:
        self.notch_filter(self.data[self.channels],freq,bandwidth)

    # return
    fig,ax = plt.subplots(1)
    ax.set_title('Filtered Ex')
    ax.semilogy(f, Pxx_den,label='Unfiltered')


    if plot_peaks:
        ax.plot(f[peaks],Pxx_den[peaks],'x')

    f, Pxx_den = signal.welch(self.data['Ex'],fs=self.sample_rate,nperseg=nperseg)
    ax.semilogy(f, Pxx_den, label='Filtered')
    ax.legend()


    ts.plot(self.data['Ex'],'r')

def detect_peaks(self,array,axis=False):
    peaks, properties = signal.find_peaks(array,threshold=1,distance=20)
    if axis:
        axis.plot(peaks,array[peaks],'kx')
    return peaks, properties