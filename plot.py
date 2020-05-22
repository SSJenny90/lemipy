import matplotlib.pyplot as plt
from scipy import signal
plt.ion()
from datetime import datetime as dt
B_LIMS = [-1000, 1000]
E_LIMS = [-8000, 8000]
import os
import export_options
import copy
import warnings

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

def welch(site,file_name=None,plot_peaks=False,save_fig=False,filter=False,decimate=False):
    nperseg = 2*10**4
    site.export_options = getattr(export_options,site.name)
    if file_name is not None:
        site.load_file(file_name)
    else:
        site.load_file(0)

    lemi = site.in_memory

    if decimate:
        lemi.decimate(decimate)

    data = lemi.data[lemi.channels].to_numpy().transpose()
    f, Pxx_den = signal.welch(data,fs=lemi.sample_rate,nperseg=nperseg)

    fig = plt.figure('Power Spectral Density')   

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')    
        fig.clear()
    
    fig, ax = plt.subplots(2,2,sharex=True, sharey=True, subplot_kw={},num='Power Spectral Density')
    t = dt.fromtimestamp(int(lemi.file_name.split('.')[0]))

    fig.suptitle('{} [{}]\n{}'.format(site.name,site.in_memory.file_name,t.strftime('%d-%b %I:%M%p')))

    if filter:
        site_copy = copy.deepcopy(site.in_memory)
        site_copy.filter(site.export_options['filter'],in_place=True)
        filtered_data = site_copy.data[lemi.channels].to_numpy().transpose()
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

    # return fig, ax
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

def lemi_log(fname):
    # fname = os.path.join('NVP','A4','F1000Hz_21_Aug_2019_1406.log')

    with open(fname) as f:
        f.readline()
        f.readline()
        f.readline()
        file_data = f.readlines()

    data = dict(
        period = [],
        num_sections = [],
        num_used = [],
    )

    for line in file_data:
        if 'Period' in line:
            data['period'].append(round(float(line.split(':')[-1]),3))

        elif 'Number of sections' in line:
            data['num_sections'].append(float(line.split(':')[-1]))

        elif 'used sections' in line:
            data['num_used'].append(float(line.split(':')[-1]))

    data = pd.DataFrame(data)
    ax = data.plot.area(stacked=False,x='period',logx=True,xlim=[data.period.min(),data.period.max()])
