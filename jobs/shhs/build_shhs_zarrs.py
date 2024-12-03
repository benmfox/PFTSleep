from pftsleep.slumber import edf_signals_to_zarr
from pftsleep.bedside import error_callback_handler

import multiprocessing as mp
from pathlib import Path
import glob
from functools import partial
import time

import pandas as pd

write_data_dir = ""
edf_files = ""
current_zarr_files = glob.glob(str(write_data_dir/"*.zarr"))

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))
# these are the files to process (havent been processed yet)
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

## NOTE THAT WHILE I DID NOT MAP HYPNOGRAMS, I DID PERFORM A MAPPING AFTER THE FACT, using the following code:
# def convert_hypnogram(hyp):
#     hyp[hyp>5] = -100
#     hyp[hyp>=4] = hyp[hyp>=4] - 1
#     return hyp
#for z in tqdm(zarr_files_shhs):
    # rt = zarr.open(z, mode='r')
    # hyp = rt['hypnogram'][:]
    # hyp = convert_hypnogram(hyp)
    # a = da.from_array(hyp, chunks='auto')
    # a.to_zarr(url=rt.store, component='hypnogram', compute=True, overwrite=True)

def main_function(file, frequency=None, write_data_dir=write_data_dir):
    try:
        _ = edf_signals_to_zarr(file, frequency=frequency, write_data_dir=write_data_dir)
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)


if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool() as pool:
        result = pool.map_async(main_function, edf_files, error_callback=error_callback_handler)
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")