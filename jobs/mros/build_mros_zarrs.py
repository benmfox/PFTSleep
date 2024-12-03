from pftsleep.slumber import edf_signals_to_zarr
from pftsleep.bedside import error_callback_handler

import multiprocessing as mp
from pathlib import Path
import glob
from functools import partial
import time
import dask.array as da

import pandas as pd

write_data_dir = ""
hyp_data_dir = ""
edf_files = ""
current_zarr_files = glob.glob(str(write_data_dir/"*.zarr"))

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))
# these are the files to process (havent been processed yet)
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

def subtract_channels(rt_grp, left_channel, right_channel, new_name):
    signal_right = rt_grp[right_channel][:]
    signal_left = rt_grp[left_channel][:]
    signal_sub = signal_left - signal_right
    a = da.from_array(signal_sub, chunks='auto')
    a.to_zarr(url=rt_grp.store, component=new_name, compute=True) # convert to zarr format
    name = mapped_label = new_name
    header = {}
    header['label'] = name
    header['mapped_label'] = mapped_label
    for i in ['digital_max', 'digital_min', 'dimension', 'prefilter', 'sample_frequency', 'sample_rate']:
        assert rt_grp[left_channel].attrs['signal_header'][i] == rt_grp[right_channel].attrs['signal_header'][i]
        header[i] = rt_grp[left_channel].attrs['signal_header'][i]
    header['physical_max'] = rt_grp[left_channel].attrs['signal_header']['physical_max'] - rt_grp[right_channel].attrs['signal_header']['physical_max']
    header['physical_min'] = rt_grp[left_channel].attrs['signal_header']['physical_min'] - rt_grp[right_channel].attrs['signal_header']['physical_min']
    rt_grp[name].attrs['signal_header'] = header

def main_function(file, frequency=None, write_data_dir=write_data_dir, hyp_data_dir=hyp_data_dir):
    try:
        rt_grp = edf_signals_to_zarr(file, frequency=frequency, write_data_dir=write_data_dir, hyp_epoch_length=30, hyp_data_dir=hyp_data_dir)
        if ('L Chin' in rt_grp.array_keys() or 'LChin' in rt_grp.array_keys()) and ('R Chin' in rt_grp.array_keys() or 'RChin' in rt_grp.array_keys()):
            # create L-R chin signal
            right_name = 'R Chin' if 'R Chin' in rt_grp.array_keys() else 'RChin'
            left_name = 'L Chin' if 'L Chin' in rt_grp.array_keys() else 'LChin'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='EMG (L-R)')
        if ('ECG L' in rt_grp.array_keys() or 'ECGL' in rt_grp.array_keys()) and ('ECG R' in rt_grp.array_keys() or 'ECGR' in rt_grp.array_keys()):
            # ecg left - right
            right_name = 'ECG R' if 'ECG R' in rt_grp.array_keys() else 'ECGR'
            left_name = 'ECG L' if 'ECG L' in rt_grp.array_keys() else 'ECGL'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='ECG (L-R)')
        if ('C4' in rt_grp.array_keys()) and ('A1' in rt_grp.array_keys() or 'M1' in rt_grp.array_keys()):
            # c4-a1
            right_name = 'A1' if 'A1' in rt_grp.array_keys() else 'M1'
            left_name = 'C4'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='C4-M1')
        if ('C3' in rt_grp.array_keys()) and ('A2' in rt_grp.array_keys() or 'M2' in rt_grp.array_keys()):
            #c3-a2
            right_name = 'A2' if 'A2' in rt_grp.array_keys() else 'M2'
            left_name = 'C3'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='C3-M2')
        if ('LOC' in rt_grp.array_keys() or 'E1' in rt_grp.array_keys()) and ('A2' in rt_grp.array_keys() or 'M2' in rt_grp.array_keys()):
            # ecg left - right
            right_name = 'A2' if 'A2' in rt_grp.array_keys() else 'M2'
            left_name = 'LOC' if 'LOC' in rt_grp.array_keys() else 'E1'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='E1-M2')
        if ('ROC' in rt_grp.array_keys() or 'E2' in rt_grp.array_keys()) and ('A1' in rt_grp.array_keys() or 'M1' in rt_grp.array_keys()):
            # eog e2-m1
            right_name = 'A1' if 'A1' in rt_grp.array_keys() else 'M1'
            left_name = 'ROC' if 'ROC' in rt_grp.array_keys() else 'E2'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='E2-M1')
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)


if __name__ == '__main__':
    #for file in edf_files:
    #    main_function(file)
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool(12) as pool:
        result = pool.map_async(main_function, edf_files, error_callback=error_callback_handler)
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")
