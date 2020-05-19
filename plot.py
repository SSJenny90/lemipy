import matplotlib.pyplot as plt
from scipy import signal
# plt.ion()
from datetime import datetime as dt
B_LIMS = [-1000, 1000]
E_LIMS = [-8000, 8000]


def save_figure(fig,fig_type,dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig('{}.png'.format(os.path.join(dir,fig_type)),format='png')

def set_lims(channel):
    if channel.startswith('E'):
        return E_LIMS
    else:
        return B_LIMS

def color(channel):
    if channel.startswith('E'):
        return 'r'
    else:
        return 'b'

def time_series(site, auto_limits=False,save_fig=False):
    
    lemi = site.in_memory

    fig,ax =  plt.subplots(len(lemi.channels),1, sharex=True)
    fig.subplots_adjust(hspace=0)

    # print(lemi.data.index)
    # lemi.data.index = lemi.data.index.tz_localize(None)
    # print(lemi.data.index)
    for axx,channel in zip(ax.flat,lemi.channels):
        axx.plot(lemi.data[channel],color(channel),linewidth=0.25,label=channel)
        axx.set_ylabel(channel, rotation=0)
        # if not auto_limits:
        #     axx.set_ylim(*set_lims(channel))

    t = dt.fromtimestamp(int(lemi.file_name.split('.')[0]))

    fig.suptitle('Time Series \n{} - {}'.format(site.name,t.strftime('%d-%b %I:%M%p')))

    if save_fig:
        save_figure(fig,'time_series',lemi.plot_directory)

def coherence(site):
    f, Cxy = signal.coherence(lemi.data['time'],lemi.data['Bx'])

def welch(site,plot_peaks=False,save_fig=False,filter=False):
    nperseg = 2*10**4
    lemi = site.in_memory
    # lemi = site
    data = lemi.data[lemi.channels].to_numpy().transpose()
    f, Pxx_den = signal.welch(data,fs=lemi.sample_rate,nperseg=nperseg)

    fig = plt.figure('Power Spectral Density')
    if fig.axes:
        xlim = fig.axes[0].get_xlim()
        ylim = fig.axes[0].get_ylim() 
    else:
        xlim = None
        
    fig.clear()
    
    fig, ax = plt.subplots(2,2,sharex=True, sharey=True, subplot_kw={},num='Power Spectral Density')
    # fig, ax = plt.subplots(2,2,sharex=True, sharey=True, subplot_kw={})
    # t = dt.fromtimestamp(int(lemi.file_name.split('.')[0]))

    # fig.suptitle('{} [{}]\n{}'.format(site.name,site.in_memory.file_name,t.strftime('%d-%b %I:%M%p')))

    if filter:
        site.in_memory.filter(site.export_options['filter'],in_place=True)
        filtered_data = site.in_memory.data[lemi.channels].to_numpy().transpose()
        filtered_f, filtered_Pxx_den = signal.welch(filtered_data,fs=lemi.sample_rate,nperseg=nperseg)

    for i,axx in enumerate(ax.flat):
        if filter:
            axx.semilogy(filtered_f, filtered_Pxx_den[i],color(lemi.channels[i]),linewidth=0.25,label='Filtered '+ lemi.channels[i])
            axx.semilogy(f, Pxx_den[i],'grey',linestyle='--',linewidth=0.25,label=lemi.channels[i])
        else:
            axx.semilogy(f, Pxx_den[i],color(lemi.channels[i]),linewidth=0.25,label=lemi.channels[i])

        # if xlim:
        #     axx.set_xlim(xlim)
        #     axx.set_ylim(ylim)



        axx.tick_params(direction='in')
        axx.legend()
        if plot_peaks:
            lemi.detect_peaks(Pxx_den[i],axis=axx)

    xlabel,ylabel = 'Frequency [Hz]','PSD [dB/Hz]'
    ax[1][0].set_xlabel(xlabel)
    ax[1][0].set_ylabel(ylabel)
    ax[1][1].set_xlabel(xlabel)
    ax[0][0].set_ylabel(ylabel)
    fig.subplots_adjust(hspace=0, wspace=0)

    if save_fig:
        save_figure(fig,'welch',lemi.plot_directory)

    # return f,Pxx_den

def spectrogram(site):
    """Not working"""
    data = site.in_memory.data
    # data,fs=lemi.sample_rate,nperseg=nperseg)
    # f,t,Sxx = signal.spectrogram(data['Bx'],fs=site.in_memory.sample_rate,nfft=1024,noverlap=50,nperseg=1024)
    fig, ax = plt.subplots(2,2,sharex=True, sharey=True, subplot_kw={})

    channels = ['Bx','By','Ex','Ey']
    for channel,axx in zip(channels,ax.flat):
        axx.specgram(data[channel], NFFT=1024, Fs=site.in_memory.sample_rate)
        axx.text(1,500,channel)
        axx.tick_params(direction='in')
        # axx.legend()

    ylabel,xlabel = 'Frequency [Hz]','Time [secs]'
    ax[1][0].set_xlabel(xlabel)
    ax[1][0].set_ylabel(ylabel)
    ax[1][1].set_xlabel(xlabel)
    ax[0][0].set_ylabel(ylabel)
