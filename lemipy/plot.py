import warnings
import copy
import export_options
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import signal
# from lemipy.lemi import 
from . import filters

plt.rcParams['figure.dpi'] = 150

# plt.ion()
B_LIMS = [-1000, 1000]
E_LIMS = [-8000, 8000]


def save_figure(fig, fig_type, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig('{}.png'.format(os.path.join(dir, fig_type)), format='png')


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


def plot(plot_type, site, file_name=None, filter=False, decimate=False):

    plots = {
        'time_series': time_series,
        'welch': welch,
        'spectrogram': spectrogram,
    }

    if not isinstance(plot_type, list):
        plot_type = [plot_type]

    for p in plot_type:
        plots[p](site, file_name, filter, decimate)


def time_series(site, file_name=None, filter=False, decimate=False):

    lemi = load(site, file_name, decimate)

    # filter data
    if filter:
        filtered = copy.deepcopy(lemi)
        filtered.filter(site.export_options['filter'], in_place=True)

    reset_figure(site, 'Time Series', decimate)

    fig, ax = plt.subplots(len(lemi.channels), 1,
                           sharex=True, subplot_kw={},num='Time Series')
    fig.subplots_adjust(hspace=0)

    # Create the figure
    for axx, channel in zip(ax.flat, lemi.channels):
        if filter:
            axx.plot(lemi.data[channel], 'grey', linewidth=0.25, label=channel)
            axx.plot(filtered.data[channel], color(
                channel), linewidth=0.25, label=channel)
        else:
            axx.plot(lemi.data[channel], color(channel),
                     linewidth=0.25, label=channel)

        axx.set_ylabel(channel, rotation=0)


def coherence(site, file_name=None, channels=[], filter=False, decimate=False):

    lemi = load(site, file_name, decimate)

    nperseg = 2*10**4

    f, Cxy = signal.coherence(lemi.data[channels[0]], lemi.data[channels[1]], fs=1000, nperseg=nperseg)


    # # perform welch on original data
    # data = lemi.data[lemi.channels].to_numpy().transpose()
    # f, Pxx_den = signal.welch(data, fs=lemi.sample_rate, nperseg=nperseg)

    # if filter:
    #     # filter data and perform welch method
    #     lemi.filter(site.export_options['filter'], in_place=True)
    #     filtered_data = lemi.data[lemi.channels].to_numpy().transpose()
    #     filtered_f, filtered_Pxx_den = signal.welch(
    #         filtered_data, fs=lemi.sample_rate, nperseg=nperseg)

    reset_figure(site, 'Coherence', decimate)

    fig, ax = plt.subplots(1, figsize=(18, 16), dpi= 80, num='Coherence')

    ax.semilogy(f, Cxy, 'b', linewidth=0.25)
    # axx.semilogy(f, Pxx_den[i], 'grey', linestyle='--',
    #                      linewidth=0.25, label=lemi.channels[i])

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Coherence')


def welch(site, file_name=None, filter=False, decimate=False):
    lemi = load(site, file_name, decimate)

    nperseg = 2*10**4

    # perform welch on original data
    data = lemi.data[lemi.channels].to_numpy().transpose()
    f, Pxx_den = signal.welch(data, fs=lemi.sample_rate, nperseg=nperseg)

    if filter:
        # filter data and perform welch method
        lemi.filter(site.export_options['filter'], in_place=True)
        filtered_data = lemi.data[lemi.channels].to_numpy().transpose()
        filtered_f, filtered_Pxx_den = signal.welch(
            filtered_data, fs=lemi.sample_rate, nperseg=nperseg)

    reset_figure(site, 'Power Spectral Density', decimate)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True,
                           subplot_kw={}, num='Power Spectral Density')

    # create the plots in for loop
    for i, axx in enumerate(ax.flat):
        if filter:
            axx.semilogy(filtered_f, filtered_Pxx_den[i], color(
                lemi.channels[i]), linewidth=0.25, label='Filtered ' + lemi.channels[i])
            axx.semilogy(f, Pxx_den[i], 'grey', linestyle='--',
                         linewidth=0.25, label=lemi.channels[i])
        else:
            axx.semilogy(f, Pxx_den[i], color(
                lemi.channels[i]), linewidth=0.25, label=lemi.channels[i])

        axx.tick_params(direction='in')
        axx.legend()

    xlabel, ylabel = 'Frequency [Hz]', 'PSD [dB/Hz]'
    ax[1][0].set_xlabel(xlabel)
    ax[1][0].set_ylabel(ylabel)
    ax[1][1].set_xlabel(xlabel)
    ax[0][0].set_ylabel(ylabel)
    fig.subplots_adjust(hspace=0, wspace=0)


def spike(site, file_name=None, channel='', index=0, filter=False):

    lemi = load(site, file_name)

    if filter:
        # filter data and perform welch method
        lemi.filter(site.export_options['filter'], in_place=True)

    reset_figure(site, 'Spike')

    despike_args = site.export_options['despike'][channel]

    data, windows, window_average, peaks = filters.despike(lemi.data[channel],fp_kwargs=despike_args)

    fig, ax = plt.subplots(1)
    ax.plot(lemi.data[channel], 'grey', linewidth=0.25, label=channel)
    ax.plot(data, 'b', linewidth=0.25, label=channel+'_despiked')
    ax.plot(lemi.data[channel].iloc[peaks], 'rx')



    fig, ax = plt.subplots(1,num='Spike')
    ax.plot(range(0,len(window_average)), window_average,'b--', label='Average signal')
    ax.plot(range(0,len(window_average)), windows[index],'r', label='Actual signal')
    # ax.plot(range(0,len(window_average)), windows[index] - window_average,'k', label='Subtracted signal')
    ax.plot(range(0,len(window_average)), windows[index] * (-signal.get_window('hamming',len(window_average))+1),'k', label='Compressed signal')

    plt.legend()


def spectrogram(site, file_name=None, filter=False, decimate=False):
    """Not working"""
    lemi = load(site, file_name, decimate)

    if filter:
        # filter data and perform welch method
        lemi.filter(site.export_options['filter'], in_place=True)

    # data,fs=lemi.sample_rate,nperseg=nperseg)
    # f,t,Sxx = signal.spectrogram(data['Bx'],fs=site.in_memory.sample_rate,nfft=1024,noverlap=50,nperseg=1024)

    reset_figure(site, 'Spectrogram', decimate)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True,
                           subplot_kw={}, num='Spectrogram')

    channels = ['Bx', 'By', 'Ex', 'Ey']
    for channel, axx in zip(channels, ax.flat):
        axx.specgram(lemi.data[channel], NFFT=1024,
                     Fs=site.in_memory.sample_rate)
                    #  Fs=decimate if decimate else site.in_memory.sample_rate)

        axx.text(1, 500, channel)
        axx.tick_params(direction='in')
        # axx.legend()

    ylabel, xlabel = 'Frequency [Hz]', 'Time [secs]'
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
        period=[],
        num_sections=[],
        num_used=[],
    )

    for line in file_data:
        if 'Period' in line:
            data['period'].append(round(float(line.split(':')[-1]), 3))

        elif 'Number of sections' in line:
            data['num_sections'].append(float(line.split(':')[-1]))

        elif 'used sections' in line:
            data['num_used'].append(float(line.split(':')[-1]))

    data = pd.DataFrame(data)
    ax = data.plot.area(stacked=False, x='period', logx=True, xlim=[
                        data.period.min(), data.period.max()])


def reset_figure(site, figure_name, decimate=''):
    # create or get an existing figure
    fig = plt.figure(figure_name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # clear the existing figure to allow replotting in same window
        # I imagine there's a much better way to do this
        fig.clear()

    # create the title
    t = dt.fromtimestamp(int(site.in_memory.file_name.split('.')[0]))
    if decimate:
        fig.suptitle('{site} [{file}] - {decimated}Hz\n{time}'.format(
            site=site.name,
            file=site.in_memory.file_name,
            decimated=decimate,
            time=t.strftime('%d-%b %I:%M%p'),
        ))
    else:
        fig.suptitle('{site} [{file}]\n{time}'.format(
            site=site.name,
            file=site.in_memory.file_name,
            time=t.strftime('%d-%b %I:%M%p'),
        ))


def load(site, file_name, decimate=None):
    
    # site.export_options = getattr(export_options, site.name)
    if file_name is not None:
        site.load_file(file_name)
    else:
        site.load_file(0)

    if decimate:
        site.in_memory.decimate(decimate)

    return site.in_memory
