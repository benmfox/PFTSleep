import json, glob, pandas as pd, numpy as np, time, multiprocessing as mp
from pathlib import Path
from pftsleep.bedside import error_callback_handler

hypnogram_tables = ""
write_data_dir = ""

map_ = {'Wake|0':0, 'Stage 1 sleep|1':1, 'Stage 2 sleep|2':2, 'Stage 3 sleep|3':3, 'Stage 4 sleep|4':3, 'REM sleep|5':4, 'Unscored|9':-100}

def main_function(idx, # index of file, not used
                  file, # list of files
                  ):
    try:
        filename = Path(file)
        df = pd.read_xml(file, xpath='.//ScoredEvent')
        df = df.loc[df['EventConcept'].isin(list(map_.keys()))] # i checked, these are the only stages
        df['shift_duration'] = df['Start'].diff().shift(-1) # need to shift to recreate duration column (some files were incorrect)
        df['num_repeats'] = df['shift_duration'] / 30 # now need to resample into 30s epochs (I checked and all durations are divisible by 30s)
        df.loc[df.num_repeats.isna(), 'num_repeats'] = df.loc[df.num_repeats.isna(), 'Duration'] / 30 # fill in the last epoch value (to the end of the edf basedo n duration). I checked and each annotation file will have exactly 1 missing shifted_duration (so fill with actual duration)
        df = df.reindex(df.index.repeat(df.num_repeats)) # reindex to labels every 30 seconds (note that the last )
        df[1] = df['EventConcept'].map(map_) # this is the default hypnigram column for writing to zarrs
        file_stem = filename.stem.strip('-nsrr') + '-hyp.csv'
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