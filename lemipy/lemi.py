import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime as dt
from functools import cached_property
from itertools import repeat

import numpy as np
import pandas as pd
import tqdm
from geopy.distance import geodesic
from scipy import signal
from tabulate import tabulate
from pprint import pprint as pp

from . import filters
sys.path.insert(0, os.path.abspath('.'))
try:
    import export_options
except ModuleNotFoundError:
    pass


PATH_TO_LEMI_EXE = 'lemimt.exe'
EXCLUDE_FOLDERS = ['__pycache__', '.vscode', 'sensors', 'lemipy', 'lemi']
TIMEZONE = ''
DECIMATE_TO = [1000, 500, 100, 50, 25]

def print_time(t):
    new_t = time.time()
    print(new_t-t)
    return new_t


def group_tail(iterator, count, tail, series=True):
    """Generates a sublist of length `count` for a given list specified by `iterator`.

    Args:
        iterator (list): The list to generate grouped items from.
        count (int): The number of items to include within each group.
        tail (int): Skip remaining files if the amount left is less than the length specified by tail.
        series (bool, optional): Converts the group to a time-indexed pandas.Series using :func:~`files_list_to_series`. Defaults to True.

    Yields:
        list|pandas.Series: Group of list items.

    .. note
        This generator will include any left over files in `iterator`. Therefore the final yielded value may or may not be equal to `count`.
    """
    itr = iter(iterator)
    while True:
        x = []
        try:
            for _ in range(count):
                x.append(next(itr))
            yield files_list_to_series(x)
        except StopIteration:
            # yields all remaining files regardless of length
            if x:
                yield files_list_to_series(x)

            # # yields the last items ONLY if they are greater than the lenth specified by tail
            # if len(x) >= tail:
            #     yield files_list_to_series(x)
            break


def group(iterator, count, series=True):
    """Generates a sublist of length `count` for a given list specified by `iterator`.

    Args:
        iterator (list): The list to generate grouped items from.
        count (int): The number of items to include within each group.
        series (bool, optional): Converts the group to a time-indexed pandas.Series using :func:~`files_list_to_series`. Defaults to True.

    Yields:
        list|pandas.Series: Group of list items.

    .. note
        This generator will skip any left over files in `iterator` if the amount of files left is less than that specified by `count`.
    """
    itr = iter(iterator)
    while True:
        try:
            if series:
                yield files_list_to_series([next(itr) for i in range(count)])
            else:
                yield [next(itr) for i in range(count)]
        except StopIteration:
            break


def list_files(directory, file_type, output='series'):
    """Generate a list of files of a certain file type from a given directory.

    Args:
        directory (str|path): A path to a directory containing files to be listed.
        file_type (str): A specific file type to search for.

    Returns:
        list: The files of specified type in the given directory.
    """
    files = [f for f in os.listdir(directory) if f.endswith(file_type)]
    if output == 'series':
        return files_list_to_series(files)
    else:
        return files


def files_list_to_series(files):
    """Convert a list of binary files to a datetime-indexed pandas series. Datetimes are determined by file names (which are Unix times) and converted to local time.

    Args:
        files (list): A list of Lemi binary files (.B423).

    Returns:
        pandas.Series: A time-index pandas series of binary files.
    """
    files = pd.Series(files, index=pd.to_datetime(
        [f.split('.')[0] for f in files], unit='s', utc=True)).sort_index()
    files.index = files.index.tz_convert(
        'Australia/Adelaide').tz_localize(None)
    if files.shape[0] > 2:
        files.freq = files.index[2]-files.index[1]
    elif files.shape[0] == 2:
        files.freq = files.index[1]-files.index[0]
    else:
        files.freq = pd.Timedelta(90, unit='m')

    return files


class Header():
    """Decodes and stores the header information of a Lemi B423 binary file.

    Attributes
    ----------
    latitude : float
        Site latitude.
    longitude : float
        Site longitude.
    lemi_number : int
        Lemi box number used to record data.
    altitude : float
        Altitude/elevation of the site.
    current : float
        Current at time of deployment.
    battery_voltage : float
        Battery voltage at time of deployment.
    deployment_time : `datetime.datetime`
        Date and time of deployment in UTC.
    coefficients : dict
        Python dict of coefficients used to determine site data.    
    """

    def __init__(self, header):
        header = header.decode('ascii').replace('%', '').split('\n')
        self.lemi_number = self.get_lemi_number(header)
        self.get_coordinates(header)
        self.altitude = self.get_altitude(header)
        self.coefficients = self.get_coefficients(header)
        self.current = self.get_current(header)
        self.battery_voltage = self.get_battery_voltage(header)
        self.deployment_time = self.get_deployment_time(header)

    def get_coefficients(self, header):
        result = {}
        for item in header[13:]:
            item = item.rstrip()
            if item:
                k, v = item.split('=')
                result[k.strip()] = float(v)
                # setattr(self,k.strip(),float(v))
        return result

    def get_lemi_number(self, header):
        return int(header[0].split('#')[-1])

    def get_deployment_time(self, header):
        date_str = header[4].split(' ')[-1] + header[5].split(' ')[-1]
        return dt.strptime(date_str.replace('\r', ''), '%Y/%m/%d%H:%M:%S')

    def get_coordinates(self, header):
        lat, direction = header[9].split(' ')[-1].split(',')
        self.latitude_dm = '{} {},{}'.format(
            lat[:2], lat[2:], direction.strip())
        self.latitude = int(lat[:2]) + round(float(lat[2:])/60, 6)
        if direction.strip() == 'S':
            self.latitude *= -1

        lon, direction = header[10].split(' ')[-1].split(',')
        self.longitude_dm = '{} {},{}'.format(
            lon[:3], lon[3:], direction.strip())
        self.longitude = int(lon[:3]) + round(float(lon[3:])/60, 6)
        if direction.strip() == 'W':
            self.longitude *= -1

    def get_altitude(self, header):
        return float(header[11].split(',')[0].split(' ')[-1])

    def get_battery_voltage(self, header):
        return float(header[6].split(' ')[-1].replace('V', ''))

    def get_current(self, header):
        return float(header[7].split(' ')[-1].replace('mA', ''))

# @pd.api.extensions.register_dataframe_accessor("lemi")
class LemiFile(Header):
    """Container for a Lemi 423 binary file

        Attributes
        ----------

        channels : list
            Channels containing recorded data. List items are auto-removed for plotting and processing if data is invalid or has not been recorded.
        sample_rate : int
            Sampling rate of the recorded data.
        data : `pandas.DataFrame`
            A pandas dataframe containing the decoded binary data. Dataframe is time indexed by the specified time + the tick interval.
        directory : str
            The directory containing the specified binary file.
        file_name : str
            Name of the binary file.
        start_time : pandas.Timestamp
            Date and time of the first recorded data point.
        finish_time : pandas.Timestamp
            Date and time of the last recorded data point.
        FILE_NAME_FORMAT : str
            Date time string formatting for output file names.
        output_filename : str
            Name of any exported file determined using the output of `LemiFile.start_time` and `LemiFile.FILE_NAME_FORMAT`
        """
    FILE_NAME_FORMAT = "%d_%b_%Y_%H%M"
    DATETIME_FORMAT = "%Y %m %d %H %M %S"

    def __init__(self, directory, file_name, data=None, sample_rate=1000, detrend=True, mag_only=False, channels=[]):
        """
        Parameters
        ----------
        file_name : str
            Name of the binary file.
        directory : str
            Directory in which to search for the specified binary file
        data : pandas.DataFrame
            A pandas dataframe from which to initialize the class (defaults to None). Useful to access LemiFile methods using a preconstructed DataFrame.
        detrend : bool
            Specify whether data is detrended on initialisation (default to True). Data may be detrended after loading using `LemiFile.detrend`.

        """
        if channels:
            self.channels = channels
        else:
            self.channels = ['Bx', 'By', 'Bz', 'Ex', 'Ey']
        self.sample_rate = sample_rate
        if data is None:
            self.data = self.load(os.path.join(
                directory, file_name), mag_only=mag_only)
        else:
            self.channels = list(data.columns)
            # if 'Bz' not in data.columns:
            # # if data.Bz.mean() == 0:
            #     self.channels.remove('Bz')
            self.data = data
        self.directory = directory
        self.file_name = file_name
        self.plot_directory = os.path.join(
            self.directory, 'plots', self.file_name)
        if detrend:
            self.detrend()

    def __repr__(self):
        return self.file_name

    @property
    def start_time(self):
        return self.data.index[0]
        # return self.data['time'][0]

    @property
    def finish_time(self):
        return self.data.index[-1]

    @property
    def output_filename(self):
        return self.start_time.strftime(self.FILE_NAME_FORMAT + ".b423.txt")

    def load(self, file_input, mag_only=False):
        """Decodes a single Lemi binary file.

        Parameters
        ----------

        file_input : str
            Either a binary (.B423) or HDF (.h5) file to be loaded.

        Returns
        -------
        df : pandas.DataFrame
            A pandas dataframe containing the decoded binary data.
        """
        if file_input.split('.')[-1] == 'B423':
            df = self._from_binary(file_input)
        elif file_input.split('.')[-1] == 'h5':
            df = self._from_hdf(file_input)
        else:
            raise ValueError(
                'Unsupported file type. Only binary and hdf files are currently supported for loading.')


        drop = [k for k in ['tick','Bx','By','Bz','Ex','Ey','CRC', 'sync', 'stage'] if k in df.keys()]
        for c in self.channels:
            if c in drop:
                drop.remove(c)

        df.drop(columns=drop, inplace=True)

        # if mag_only:
        #     df.drop(columns=['Ex', 'Ey'], inplace=True)
        #     [self.channels.remove(channel) for channel in ['Ex', 'Ey']]

        return df

    def detrend(self):
        """Remove the mean and linear trend from all data channels."""
        detrended = signal.detrend(
            self.data[self.channels], type='linear', axis=0)
        detrended = signal.detrend(detrended, type='constant', axis=0)
        self._replace_columns(detrended)

    def _replace_columns(self, array, channels=None):
        """For some reason this is waaaay faster (1 second) than assigning an array to multiple dataframe columns using df[column_list] = array (~3.5 seconds)."""
        if not channels:
            channels = self.channels
        elif isinstance(channels, str):
            self.data.loc[:, channels] = array
            return

        if array.ndim > 1 and not array.shape[1] == len(channels):
            raise ValueError(
                'Width of array must equal number of columns being assigned to. Array shape: {}'.format(array.shape))

        for i, channel in enumerate(channels):
            # print('Replacing data in {}'.format(channel))
            self.data.loc[:, channel] = array[:, i]

    def filter(self, filter, target=None, in_place=False):
        """Apply a filter to the dataset

        Parameters
        ----------
        filter : list of tuples | dict
            The filter to be applied to the data channels specified by target.
        target : str | list of strings, optional
            Channel or list of channels to apply filter to. Ignored if a dict is supplied as the filter.
        in_place : bool
            If True, applies the filter directly to the dataset. Otherwise returns the filtered channels as an array.

        Return
        ------
        y : ndarray | None
            If in_place is False, returns the filtered signal, else returns None

        Examples
        --------
        Apply the same comb filter to all channels within the dataset 
        >>> filters = [
                (50,6),
                (150,8),
                (250,8),
                (350,8),
                (450,8),
            ]
        >>> site.filter(filters,in_place=True)

        or apply to only the magnetix channels...

        >>> site.filter(filters, target=['Bx','By'], in_place=True)

        Apply individual filters to specific channels

        >>> filters = {}
        >>> filters['Bx'] = [
                (50,6),
                (150,8),
            ]
        >>> filters['Ex'] = [
                (250,8),
                (350,8),
            ]
        >>> site.filter(filters,in_place=True)
        """
        channels = [c for c in self.channels if c not in ['RBx', 'RBy']]
        array = []
        for key, filt in filter.items():
            b, a = filters.comb(filt, self.sample_rate)

            # None of these filters were valid for the given export options
            if b is None and a is None:
                continue

            to_be_filtered = [c for c in channels if c.startswith(key) or c.endswith(
                key.lower()) or c in key.split('_') or key == 'all']

            if in_place:
                self._apply_filter(b, a, to_be_filtered, in_place)
            else:
                for channel in to_be_filtered:
                    array.append(self._apply_filter(b, a, channel))

        return array

    def despike(self, channel, prominence, distance, window_size=50, in_place=False):

        channels = [c for c in self.channels if c not in ['RBx', 'RBy']]
        array = []
        for key, filt in filter.items():
            to_be_despiked = [c for c in channels if c.startswith(key) or c.endswith(
                key.lower()) or c in key.split('_') or key == 'all']

            for channel in to_be_filtered:
                self.data[channel], _, _ = filters.despike(self.data[channel], prominence, distance, window_size)

    def decimate(self, decimate_to):
        """Decimate the data to sampling frequency specified by decimate_to using `scipy.signal.decimate`.

        Parameters
        ----------
        decimate_to : int
            The desired output sampling frequency. Determines the downsampling factor for `scipy.signal.decimate`.

        Returns
        -------
        decimated : pd.DataFrame
            A new dataframe decimated to decimate_to. This is the same value as accessing `LemiFile.data` immediately after preforming decimation.

        Notes
        -----
        Decimation is always applied directly to the dataframe.
        """
        if decimate_to >= self.sample_rate:
            raise ValueError(
                'Target sampling frequency must be less than the sampling frequency of the original data. Original sampling frequency calculated to be {}Hz'.format(self.sample_rate))

        if self.sample_rate % decimate_to != 0:
            raise ValueError(
            "The target decimation must be an exact divisor of the data sampling rate."
        )

        q = int(round(self.sample_rate/decimate_to))  # downsampling factor

        # apply proper decimation on the data columns
        decimated_array = signal.decimate(self.data[self.channels],
                                          q=q,
                                          ftype='fir',
                                          axis=0)

        self.data = self.data.iloc[::q, :].copy()
        self._replace_columns(decimated_array)
        self.sample_rate = decimate_to
        return self.data

    def export(self, file_name='', prepend_filename='', export_format=None):
        """Export the current file as a .b423.txt file used in the LemiGraph software.

        Parameters
        ----------
        out_directory : str, optional
            The output directory (defaults to same directory as the binary file).
        file_name_prepend : str, optional
            String to prepend to the exported file name. Useful to distinguish decimated or filtered data exports.
        """

        if prepend_filename:
            prepend_filename += '_'

        if export_format is None:
            if file_name:
                file_out = os.path.join(
                    self.directory, prepend_filename+file_name)
            else:
                file_out = os.path.join(
                    self.directory, prepend_filename+self.output_filename)

            fmt = []
            for channel in self.data.columns:
                if channel.startswith('B') or channel.startswith('R'):
                    fmt.append("%.4f")
                elif channel.startswith('E'):
                    fmt.append("%.3f")

            data = self.data.to_numpy()
            with open(file_out, 'w') as f:
                # new - ~ 9seconds
                fmt = ' '.join(fmt)
                fmt = '\n'.join([fmt]*data.shape[0])
                data = fmt % tuple(data.ravel())        
                f.write(data)
            
        elif export_format == 'hdf':
            file_out = os.path.join(
                self.directory, file_name.split('.')[0]+'.h5')
            # print('Exporting file {}'.format(file_out))

            self.data.to_hdf(file_out, key='data')


        return file_out, list(self.data.columns)

    def _variance(self):
        return self.data[self.channels].var()

    def _apply_filter(self, b, a, target=None, in_place=False):
        if not target:
            target = [channel for channel in self.channels if channel not in [
                'RBx', 'RBy', 'RBz']]

        filtered_signal = signal.filtfilt(b, a, self.data[target], axis=0)

        if in_place:
            self._replace_columns(filtered_signal, channels=target)
        else:
            return filtered_signal

    def _from_binary(self, binary_file):
        binary_format = np.dtype([
            ('time', 'u4'),
            ('tick', 'u2'),
            ('Bx', 'i4'),
            ('By', 'i4'),
            ('Bz', 'i4'),
            ('Ex', 'i4'),
            ('Ey', 'i4'),
            ('sync', 'b'),
            ('stage', 'B'),
            ('CRC', 'i2'), ])

        with open(binary_file, 'rb') as f:
            super().__init__(f.read(1024))
            data = np.fromfile(f, dtype=binary_format)

        df = pd.DataFrame(data)
        # df['time'] = pd.to_datetime(df['time'],unit='s',utc=True) + pd.to_timedelta(df['tick'],unit='ms')
        df['time'] = pd.to_datetime(
            df['time'], unit='s') + pd.to_timedelta(df['tick'], unit='ms')

        # df['time_delta'] = df['time'] - df['time'][0]
        # df['minutes_elapsed'] = df['time_delta'].dt.total_seconds()/60

        df['Bx'] = df['Bx'] * self.coefficients['Kmx'] + self.coefficients['Ax']
        df['By'] = df['By'] * self.coefficients['Kmy'] + self.coefficients['Ay']
        df['Bz'] = df['Bz'] * self.coefficients['Kmz'] + self.coefficients['Az']
        df['Ex'] = df['Ex'] * self.coefficients['Ke1'] + self.coefficients['Ae1']
        df['Ey'] = df['Ey'] * self.coefficients['Ke2'] + self.coefficients['Ae2']

        df.set_index('time', inplace=True)
        if TIMEZONE:
            # df.index = df.index.tz_convert(TIMEZONE).tz_localize(None)
            pass
        # drop = ['tick', 'CRC', 'sync', 'stage']  # always dropped
        self.sample_rate = df['tick'].max()+1



        # if df['Bz'].mean() < -5000 or df['Bz'].var() < 10:
        #     drop.append('Bz')
        #     # avoids computationally expensive processing on Bz if no data is recorded
        #     self.channels.remove('Bz')

            # if mag_only:
            #     drop.append('Bz')
            # else:
            #     df['Bz'] = np.zeros(df.shape[0],dtype=int)


        return df

    def _from_hdf(self, hdf_file):
        return pd.read_hdf(hdf_file)


class Site(Header):
    """Represents an individual site in the current survey

    Attributes
    ----------
        name : str
            Name of the site.
        directory : str
            Directory containing site data.
        export_options : dict
            A python dict specifying filters and decimation parameters to use during export.
        coordinates : tuple
            Site coordinates in the form (lat,lon).
        files : list
            A list of binary files within the `Site` directory.
        pickup_time : datetime.datetime
            Time the site was picked up. Requires loading a binary file when first accessed. Attribute is cached so that all subsequent queries do not require loading.
        sample_rate : int
            Sampling frequency of the instrument. Requires loading a binary file when first accessed. Attribute is cached so that all subsequent queries do not require load a file.
    """

    def __init__(self, site_name, survey_data=None, directory=os.getcwd(), load=False, remote=None, channels=[]):
        """
        Parameters:

            name : str
                Site name (case sensitive).
            field_data : dict | pandas `Series`
                If provided, deployment field data such as line length and azimuth collected during the survey will be saved as attributes on the object. 
            directory : str, opt
                Specifies the directory containing site data in binary (.B423)format. By default, the `Site` object will search for files in a subdirectory specified by `Site.site_name` within the survey folder.
            load : bool | str | int
                Whether a file is loaded on initialisation. If True, the first file in the directory will be loaded. A specific file can be loaded by providing the file name or integer index of `Site.files`.

        .. note:: The `Site` object subclasses the `Header` object and therefore all `Header` attributes are available directly from the `Site` object. e.g `Site.latitude` will return the `latitude' attribute of the `Header` object. See `Header` for available attributes'
        """
        self.name = site_name
        self.directory = os.path.join(directory, site_name)
        self.get_field_data(survey_data)

        with open(os.path.join(self.directory, self.files[0].split('.')[0]+'.B423'), 'rb') as first:
            super().__init__(first.read(1024))

        self.deployment_length = self.pickup_time - self.deployment_time
        self.filters = None
        self.remote = remote
        self.file_freq = self.files.index[2] - self.files.index[1]       
        self.channels = channels

        try:
            self.export_options = getattr(export_options,self.name)
        except Exception:
            pass
        self.export_options = {}

        if load is True:
            self.load_file(self.files[0])
        elif load is not False:
            self.load_file(load)

    def __str__(self):
        return str(pd.Series(self.summary()))
        # return ''
        
        # return tabulate([[k, v] for k, v in self.summary().items()])

    @property
    def coordinates(self):
        return (self.latitude, self.longitude)

    @property
    def files(self):
        files = list_files(self.directory, '.B423')
        return files

    @cached_property
    def in_memory(self):
        # # return self.load_file
        # if not self._in_memory:
        #     self.load_file(self.files[0])
        # return self._in_memory
        return self.load_file(self.files[0])

    @cached_property
    def pickup_time(self):
        last_file = os.path.join(self.directory, self.files[-1])
        with open(last_file, 'rb') as last:
            last.seek(-30, 2)
            return dt.utcfromtimestamp(int.from_bytes(last.read(4), byteorder=sys.byteorder))

    @cached_property
    def sample_rate(self):
        return self.in_memory.sample_rate

    def remote_files(self, site_files=None, skip_first=False, skip_last=False):
        if site_files is None:
            site_files = self.files

        first = site_files.index[0] if not skip_first else site_files.index[1]
        last = site_files.index[-1] if not skip_last else site_files.index[-2]

        start = first-self.file_freq
        finish = last+self.file_freq

        result = {f: list_files(remote.directory, '.h5')[start:finish] for f, remote in self.remote.items(
        ) if not list_files(remote.directory, '.h5')[start:finish].empty}

        return pd.concat(result.values(), keys=result.keys())

    def get_field_data(self, survey_data):
        if survey_data is None:
            return
        else:
            data = survey_data.loc[self.name]
        self.Ex = {k.split('_')[-1]: v for k,
                   v in data.items() if k.startswith('Ex')}
        self.Ey = {k.split('_')[-1]: v for k,
                   v in data.items() if k.startswith('Ey')}

    def summary(self, detailed=False):
        """Return a brief summary of the current site.

        Parameters:
            detailed : bool
                If True, return statistical information calculated for each data channel. Requires loading a file into memory.
        """
        exclude = ['coefficients', 'fileout_format', 'directory', 'in_memory',
                   'filters', 'latitude_dm', 'longitude_dm', 'Ex', 'Ey', 'remote']
        output = {k: v for k, v in self.__dict__.items() if k not in exclude}
        if detailed:
            output.update({k+'_var': int(v) for k, v in dict(self.in_memory._variance()).items()})
        return {**output, 'distance_to_remote': self.distance_to_remote()}

    def load_file(self, binary_file=None, kwargs={}):
        """Loads a binary file into a `LemiFile` instance and stores in `Site.in_memory`. `Site.sample_rate` is also set to avoid future calls to load file when accessing the `sample_rate` attribute.
        """
        if isinstance(binary_file, int):
            # find the file at the given index
            binary_file = self.files[binary_file]
        elif isinstance(binary_file, list):
            # find the file between to times
            binary_file = self.files.between_time(*binary_file)[0]
        
        elif binary_file.split('.')[-1] not in ['B423','h5']: #it's a string but not a filename
            # assum it's a time so find the next file after the given time
            hours = int(binary_file[:2])
            binary_file = self.files.between_time(
                binary_file,'{}{}'.format(hours+2,binary_file[2:]))[0]

        if 'channels' not in kwargs.keys():
            kwargs['channels'] = self.channels


        self.in_memory = LemiFile(self.directory, binary_file, **kwargs)
        self.sample_rate = self.in_memory.sample_rate

        return self.in_memory

    def decimate(self, site_files, decimate_to, remote_files=None, save=True):
        """Decimate files using `scipy.signal.decimate` and combine into one dataframe.

        Parameters
        ----------

            site_files : list | pd.Series
                Site files to be decimated and combined into one dataframe
            decimate_to : int
                Desired output sampling frequency
            remote_files : list | pd.Series
                Remote files from which to source replacement magnetic channels
            save : bool, opt
                If True, will export the decimated `pd.DataFrame` on completion'

        Returns
        -------
        data : `LemiFile` instance
            A new LemiFile instance holding the decimated data from the combined input site_files
        """
        # merge the site data into a single dataframe
        data = pd.concat(
            [self.load_file(f).data for f in site_files], copy=False)

        # determine the available magnetic channels from the site data
        mag_channels = [
            channel for channel in data.columns if channel.startswith('B')]

        # get the approriate remote Site instance for the given remote_files
        if remote_files is not None:
            remote = self.remote.get(remote_files.index.unique(0)[0])

            # combine the remote files into a single dataframe
            remote_data = pd.concat([
                remote.load_file(
                f, {'mag_only': True, 'detrend': False}).data[mag_channels] for f in remote_files], copy=False)

            # rename the maganetic channels to avoid a name conflict with the site magnetic channels
            remote_data.rename(
                columns={ch: 'R'+ch for ch in mag_channels}, inplace=True)

            if self.export_options.get('remote_offset', None):
                remote_data.index = remote_data.index.shift(
                    self.export_options.get('remote_offset'), 's')

            # merge site and remote data into a single dataframe according to the time index of the site data
            data = pd.concat([data, remote_data], axis=1,
                             join='inner', copy=False)

            del remote_data

        # convert merged data to a LemiFile instance
        data = LemiFile(self.directory, file_name='{}Hz'.format(
            decimate_to), data=data)

        # decimate the merged data
        if decimate_to != 1000:
            data.decimate(decimate_to)

        if save:
            data.export()
        return data

    def export(self, files, decimate_to=None, create_edi=True, remote=True):
        
        # Find corresponding remote files if required
        if remote and self.export_options.get('remote', True):
            remote_files = self.remote_files(site_files=files)

            # ALL REMOTE FILES MUST COME FROM THE SAME REMOTE FOLDER
            if len(remote_files.index.unique(0)) != 1:
                return
            else:
                remote = self.remote.get(remote_files.index.unique(0)[0])
        else:
            remote_files = None

        date_from_file = dt.fromtimestamp(
            int(files[0].split('.')[0])).strftime("%d_%b_%Y_%H%M")

        output_filename = '{}{}Hz_{}.txt'.format(
            'F' if self.export_options.get('filter') else '',
            decimate_to,
            date_from_file)

        print('Processing {}'.format(output_filename))

        output = pd.DataFrame()

        for f in files:

            # gets the available magnetic channels on the site
            # use these to get appropriate data from remote
            mag_channels = [c for c in self.channels if c.startswith('B')]
            
            data = self.load_file(f,kwargs={'channels':self.channels}).data

            # determine the available magnetic channels from the site data
            # mag_channels = [channel for channel in data.columns if channel.startswith('B')]

            if remote_files is not None:

                # combine the remote files into a single dataframe
                remote_data = pd.concat(
                    [remote.load_file(
                        f, kwargs={'mag_only': True, 'detrend': False}).data[mag_channels] for f in remote_files[:2]], copy=False)

                # rename the magnetic channels to avoid a name conflict with the site magnetic channels
                remote_data.rename(
                    columns={ch: 'R'+ch for ch in mag_channels}, inplace=True)

                if self.export_options.get('remote_offset', None):
                    remote_data.index = remote_data.index.shift(
                        self.export_options.get('remote_offset'), 's')

                # merge site and remote data into a single dataframe according to the time index of the site data
                data = pd.concat([data, remote_data], axis=1,
                                 join='inner', copy=False)

                # remove from memory
                del remote_data

                # remove first file from remote_files
                remote_files.drop(remote_files.index[0], inplace=True)

            # convert merged data to a LemiFile instance
            data = LemiFile(
                directory = self.directory, 
                file_name='{}Hz'.format(decimate_to), data=data)

            # Decimate the merged data
            if decimate_to != 1000:
                data.decimate(decimate_to)

            output = pd.concat([output, data.data], copy=False)

        data = LemiFile(
            directory = self.directory, 
            file_name='{}Hz'.format(decimate_to), 
            data=output, 
            sample_rate=decimate_to)

        del output

        # Apply filters
        if self.export_options.get('filter'):
            data.filter(self.export_options['filter'], in_place=True)

        # Export
        if self.export_options.get('format') == 'hdf':
            exported_file_name, channels = data.export(
                file_name=f, export_format='hdf')
        else:
            exported_file_name, channels = data.export(
                file_name=output_filename)

        if create_edi and not self.export_options.get('format') == 'hdf':
            self.to_edi(exported_file_name, channels)

    def batch_export(self, processes=None, test=False):
        print('Starting batch export of site "{}"\n'.format(self.name))
        start = dt.now()
        if not processes:
            processes = int(multiprocessing.cpu_count()/2)


        if self.export_options.get('between_time'):
            between_time = self.export_options.get('between_time')
            files_list = self.files[between_time[0]:between_time[1]]
        else:
            files_list = self.files

        export_args = []

        for decimate_to in sorted(list(self.export_options.get('decimate', DECIMATE_TO))):
            num_files = int(1000/decimate_to)
            export_args += list(zip(group_tail(files_list,num_files*2,num_files),repeat(decimate_to)))
            # export_args += list(zip(group_tail(self.files,
                                            #    num_files, num_files), repeat(decimate_to)))

        if test:
            self.export(self.files[2:3], 1000)
        else:
            with multiprocessing.Pool(processes=processes) as pool:
                pool.starmap(self.export, export_args)

        print('Export finished in {}'.format(
            dt.now() - start
        ))

    def create_config(self, file_name, channels):
        # config_file = os.path.join(file_name.split('.')[0])
        # config_file = os.path.join(self.name,file_name.split('.')[0])
        fname = os.path.split(file_name)[1]
        try:
            sample_rate = int(fname.split('Hz')[0])
        except ValueError:
            sample_rate = int(fname.split('Hz')[0][1:])

        with open(file_name.replace('.txt', '.cfg'), 'w') as f:
            f.write(
                'SITE {name}\n'
                'LATITUDE {latitude}\n'
                'LONGITUDE {longitude}\n'
                'ELEVATION {elevation}\n'
                'DECLINATION {declination}\n'
                'SAMPLING {sample_rate}\n'
                'NCHAN {number_of_channels}\n'
                .format(
                    name=file_name.split('.')[0],
                    latitude=self.latitude_dm,
                    longitude=self.longitude_dm,
                    elevation=self.altitude,
                    declination=0,
                    sample_rate=1/sample_rate,
                    number_of_channels=len(channels),))

            for i, channel in enumerate(channels):
                if channel.startswith('B') or channel.startswith('R'):
                    f.write('  {i}   {H:.7f} 1  {Hrsp}\n'.format(
                        i=i+1,
                        H=.001,
                        Hrsp='l120new.rsp'))

                elif channel.startswith('E'):
                    E = getattr(self, channel)['length'] ** -1
                    f.write('  {i}   {E:.7f} 1  {Ersp}\n'.format(
                        i=i+1,
                        E=E if getattr(self, channel)['azimuth'] not in [
                            0, 90] else E*-1,
                        Ersp='e000.rsp'))

            if len(channels) > 5:
                f.write('NREFCH        2\n')

    def to_edi(self, file_name, channels, sensor_response=True, robust=True, coherence_presorting=False, prewhitening=None, delete_on_completion=False, suppress_lemi_output=False):
        # file_name = os.path.join('NVP','D1',file_name)
        self.create_config(file_name, channels)

        command = ['lemimt.exe']

        if sensor_response:
            command.append('-f')
        if robust:
            command.append('-r')
        if coherence_presorting:
            command.append('-c')
        if prewhitening:
            command.append('-w{}'.format(prewhitening))

        command.append(file_name)

        if self.export_options.get('clean', True):
            subprocess.Popen(command, shell=False,
             stdin=None, stdout=subprocess.DEVNULL, stderr=None, close_fds=True)
        else:
            # when clean is False, lemimt.exe output will be saved to file for later viewing
            # this dramatically slows lempiy as it will wait for this subprocess to finish so it can capture it's output
            log_file = file_name.split('.')[0]+'.log'
            with open(log_file, 'w') as f:
                subprocess.run(command, stdout=f)


        # if delete_on_completion or self.export_options.get('clean', True):
        #     os.remove(file_name)
        #     os.remove(file_name.replace('.txt', '.cfg'))

    def distance_to_remote(self):
        return round(geodesic(self.coordinates, list(self.remote.values())[0].coordinates).km, 2)


class Remote(Site):

    def __init__(self, site_name, folders, survey_data=None, load=False, remote=None):
        self.folders = {remote_folder: Site(
            remote_folder, survey_data) for remote_folder in folders}
        self.name = site_name
        super().__init__('', survey_data, directory=list(
            self.folders.values())[0].directory, load=load, remote=remote)

    @property
    def files(self):
        result = {f: list_files(remote.directory, '.B423') for f, remote in self.folders.items(
        ) if not list_files(remote.directory, '.B423').empty}
        return pd.concat(result.values(), keys=result.keys())

    def distance_to_remote(self):
        return 0

    def summary(self):
        summary = super().summary()
        del summary['distance_to_remote']
        del summary['pickup_time']

        return {**summary, 'remote_folders': len(self.folders)}


class Survey():
    """ Survey object that summarises an entire object and stores individual site summaries.

    Attributes
    ----------
        directory : str
            The working directory.
        remote : str
            Name of the remote site used in the survey.
        survey_data : pd.DataFrame()
            A pandas dataframe of deployment data. Required for processing of sites via lemimt.exe.
        site_list : list
            Populated through introspection of the working directory. All folders that contain any lemi binary files are added to the list.
        sites : dict
            Python dict of `Site` objects used to access individual site data.
        count : int
            Number of sites
    """

    def __init__(self, directory=os.getcwd(), remote='/', survey_data=None, channels=[]):
        """        
        Parameters:
            directory : str, opt
                Path to the directory containing site folders and data (defaults to current working directory). 
            remote : str, opt
                Name of the site used as remote (defaults to `/` to ensure no effect on os.path calls if left blank).
            survey_data : str, opt
                Path to a .csv file containing reported data at each site. Processing will be unavailable if not specified as this information cannot be determined through introspection of binary files.
        """
        self.directory = directory
        self.channels = channels

        if survey_data is not None:
            self.survey_data = pd.read_csv(survey_data, index_col='name')
        else:
            self.survey_data = None

        # self.site_list = self.populate_site_list()
        self.remote = self.get_remote(remote)
        self.sites = self.get_sites()
        self.init_export_options()

    def __str__(self):
        return tabulate([[k.replace('_', ' ').title(), v] for k, v in self.summary().items()])

    @property
    def site_list(self):
        return [name for name in sorted(os.listdir(self.directory)) if os.path.isdir(os.path.join(self.directory, name)) and not list_files(os.path.join(self.directory, name), '.B423').empty]

    @property
    def count(self):
        return len(self.sites)

    def get_remote(self, remote_name):
        if remote_name != '/':
            # find all site folders that startwith remote_name
            return {site: Site(site, survey_data=self.survey_data, directory=self.directory, channels=self.channels) for site in self.site_list if site.startswith(remote_name)}

    def get_sites(self):
        return {site: Site(site,
                           survey_data=self.survey_data,
                           remote=self.remote,
                           directory=self.directory,
                           channels=self.channels) for site in self.site_list}

    def dataframe(self, filename=None, detailed=False, processes=None):
        """Returns a pandas dataframe summarising the survey. If a filename is provided, the dataframe will be saved to file using the pandas `DataFrame.to_csv' method.

        Parameters:

            filename : str, opt
                File name of the exported csv file.
            detailed : bool, opt
                If True, will determine statistics for each site to help in determining data quality. Requires loading individual site data so can take a long time depending on the number of sites. Utilizes multiprocessing to speed things up (defaults to False)
            processes : int, opt
                If detailed is True, specify the number of cpu's to use. Defaults to half the cpu's as determined by `multiprocessing.cpu_count`.
        """
        if not self.sites:
            self.get_sites()

        if detailed:
            self.site_list = []

            if not processes:
                processes = int(multiprocessing.cpu_count()/2)

            with multiprocessing.Pool(processes=processes) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(self._get_detailed_site_summary, self.sites.values()), total=len(self.sites.values())):
                    pass

        else:
            x = [site.summary(detailed)
                              for site in self.sites.values()]

        df = pd.DataFrame(x)
        df.sort_values(by=['name'], inplace=True)
        df.set_index('name', inplace=True)
        if filename:
            df.to_csv(filename)
        return df

    def summary(self):
        """Provides a brief summary of the survey."""
        print(self.directory.capitalize())
        df = self.dataframe()
        return dict(
            site_count=df.shape[0],
            lat_bounds=(df.latitude.min(), df.latitude.max()),
            lon_bounds=(df.longitude.min(), df.longitude.max()),
            first_deployment=df.deployment_time.min(),
            last_pick_up=df.pickup_time.max(),
            survey_duration=df.pickup_time.max() - df.deployment_time.min(),
            # index out dedicated remote to get more accurate average
            # ave_deployment_length = df[~df.index.str.contains(self.remote)].deployment_length.mean(),
            lemi_boxes_used=df.lemi_number.sort_values().unique(),
        )

    def _get_detailed_site_summary(self, site):
        self.site_list.append(self.sites[site].summary(detailed=True))

    def init_export_options(self):
        fname = os.path.join(self.directory,'export_options.py')
        if not os.path.exists(fname):
            with open(fname,'w') as f:
                f.write("from lemipy.filters import remove_50hz_odd, remove_50hz_even, remove_signal\n\n")
                f.write(
                    "# This module was automatically created during initialisation of a new survey.\n" + 
                    "# It contains the outline needed to define some export options for you to use with the lemipy package.\n" +
                    "# All options that are applied to a particular site in this file will be used during any bulk processing using the lemipy package.\n" +
                    "# This file serves as not only a list of instructions but also a record of what processing has been applied to each site so that coming back and picking up where you left of is simple.\n" +
                    "# For a list of all possible options for each site, please see the documentation.\n\n"
                )
                f.write("# TODO - Ex looks noisy, B channels are fine, strong powerline hum and harmonics (example comment)\n")
                for site in self.site_list:
                    if len(site.split(' ')) > 1:
                        continue
                    f.write("{} = {{\n".format(site))
                    f.write("    'filter': {\n\n")
                    f.write("        },\n")
                    f.write("    'remote_offset': 0,\n")
                    f.write("    }\n\n")

            print('Warning: You will need to load the lemipy module again before any export_options are included on your survey object.')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # freeze_support()
    # t=time.time()
    survey = Survey(directory='NVP', remote='C5', survey_data='survey_data.csv')
    site = survey.sites['B3']
