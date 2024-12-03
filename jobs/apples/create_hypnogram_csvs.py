import json, glob, pandas as pd, numpy as np, time, multiprocessing as mp
from pathlib import Path
from pftsleep.bedside import error_callback_handler

hypnogram_tables = ""
write_data_dir = ""

map_ = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}

def main_function(idx, # index of file, not used
                  file, # list of files
                  ):
    try:
        filename = Path(file)
        df = pd.read_table(file)
        df = df.loc[df['class'].isin(['W','N1','N2','N3','R'])] # i checked, these are the only stages, and all stages are aligned and no gaps
        df[1] = df['class'].map(map_) # this is the default hypnigram column for writing to zarrs
        file_stem = filename.stem + '-hyp.csv'
        df.to_csv(write_data_dir/Path(file_stem), index=False)
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)
    

if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool() as pool:
        result = pool.starmap_async(main_function, enumerate(hypnogram_tables), error_callback=error_callback_handler)
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")