import json, glob, pandas as pd, numpy as np, time, multiprocessing as mp
from pathlib import Path
from pftsleep.bedside import error_callback_handler

hypnogram_jsons = ""
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
def main_function(idx, # index of file, not used
                  file, # list of files
                  ):
    try:
        filename = Path(file)
        with open(file) as f:
            hypnogram_json = json.load(f)
        hyp_data = np.array(hypnogram_json['Data']['10sEpochs'], dtype=np.int64).repeat(10)
        epochs = np.array(range(1, len(hyp_data) + 1), dtype=np.int64)
        df = pd.DataFrame(zip(epochs, hyp_data), columns = [0,1])
        file_stem = filename.stem + '-hyp.csv'
        df.to_csv(filename.parent/Path(file_stem), index=False)
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)
    

if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool() as pool:
        result = pool.starmap_async(main_function, enumerate(hypnogram_jsons), error_callback=error_callback_handler)
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")