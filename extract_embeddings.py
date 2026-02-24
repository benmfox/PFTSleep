import torch, pandas as pd, os, glob, lightning.pytorch as pl

from pftsleep.train import PatchTFTSimpleLightning
from pftsleep.slumber import SelfSupervisedTimeFrequencyDataset, ALL_FREQUENCY_FILTERS, VOLTAGE_CHANNELS, ALL_CHANNELS
from pftsleep.inference import (ENCODER_DEFAULTS, 
    FREQUENCY_DEFAULT, 
    HYPNOGRAM_EPOCH_SECONDS_DEFAULT, 
    HYPNOGRAM_FREQUENCY_DEFAULT, 
    HYPNOGRAM_PADDING_DEFAULT, 
    SEQUENCE_LENGTH_SECONDS_DEFAULT,
    MEDIAN_FILTER_KERNEL_SIZE_DEFAULT)

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(True)

encoder_path = ''
embedding_file_name = 'shhs-embeddings.pt'
zarr_file_path = ''

zarr_files  = glob.glob(os.path.join(zarr_file_path, "*.zarr/"))
channels = ALL_CHANNELS
arch = ENCODER_DEFAULTS
BATCHSIZE = 16
num_workers = 4
n_gpus = 1

# train model
if __name__ == "__main__":
    ds = SelfSupervisedTimeFrequencyDataset(zarr_files=zarr_files,
                                            channels=channels, 
                                            frequency=FREQUENCY_DEFAULT,
                                            trim_wake_epochs=False,
                                            return_hypnogram_every_sec=HYPNOGRAM_EPOCH_SECONDS_DEFAULT,
                                            hypnogram_frequency=HYPNOGRAM_FREQUENCY_DEFAULT,
                                            hypnogram_padding_mask=HYPNOGRAM_PADDING_DEFAULT,
                                            scale_channels=False, # we could check tuning on this
                                            start_offset_sec=0,
                                            clip_interpolations=None, # we could tune this
                                            include_partial_samples=True, 
                                            return_sequence_padding_mask=True,
                                            butterworth_filters=ALL_FREQUENCY_FILTERS, # dictionary of low pass, high pass, and bandpass dictionary to perform on channels
                                            median_filter_kernel_size=MEDIAN_FILTER_KERNEL_SIZE_DEFAULT, # if not none, will apply median filter with kernel size
                                            voltage_channels=VOLTAGE_CHANNELS, # if not None, these channels units will be looked at and changed to microvolts from mv uv etc.
                                            max_seq_len_sec=SEQUENCE_LENGTH_SECONDS_DEFAULT, 
                                            sample_seq_len_sec=SEQUENCE_LENGTH_SECONDS_DEFAULT, 
                                            sample_stride_sec=SEQUENCE_LENGTH_SECONDS_DEFAULT)

    sample_df = ds.sample_df.copy()

    data_loader = DataLoader(ds, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    
    
    pftsleep_encoder = PatchTFTSimpleLightning.load_from_checkpoint(encoder_path, map_location='cpu')
    
    trainer = pl.Trainer(precision="32",
                     enable_checkpointing=True,
                     enable_progress_bar=True,
                     enable_model_summary=True, 
                     strategy="ddp",
                     log_every_n_steps=50,
                     num_sanity_val_steps=0, 
                     accelerator="gpu", 
                     devices=n_gpus, 
                     fast_dev_run=False)
    
    preds = trainer.predict(model=pftsleep_encoder, train_dataloaders=data_loader, val_dataloaders=None, ckpt_path=None)
    preds = torch.cat(preds)
    torch.save(preds, f"{embedding_file_name}")
    sample_df.to_csv(f"{embedding_file_name.replace('.pt', '.csv')}", index=False)

