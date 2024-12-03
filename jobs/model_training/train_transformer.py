import torch, pandas as pd, wandb, os, glob, lightning.pytorch as pl

from pftsleep.train import PatchTFTSimpleLightning
from pftsleep.slumber import SelfSupervisedTimeFrequencyDataset, ALL_FREQUENCY_FILTERS, VOLTAGE_CHANNELS
from pftsleep.loss import mse, mae, cosine_similarity

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import random

from pathlib import Path

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(True)

models_dir = ""
data_dir = ""
model_run = ''
name = ''

use_sequence_padding_mask = return_sequence_padding_mask = True
trim_wake_epochs = True if return_sequence_padding_mask else False # if not returning a padding mask, signals will be set to 0, idk whats better
return_hypnogram_every_sec = 30
hypnogram_frequency = 1
hypnogram_padding_mask = -100

val_check_interval=1.0 # 1.0 is the default (each epoch), use interger for steps, fraction for fraction of epoch

use_gradient_clipping = False
gradient_clip_val = 1
gradient_clip_algorithm = 'norm' if use_gradient_clipping else None

BATCHSIZE = 3
accumulate_grad_batches = 8
EPOCHS = 100
n_gpus = -1
num_workers = 8
learning_rate = 1e-4
loss_func = 'MSE'
scale_channels = False # setting to false bc of out of distribution data - revin should handle this
random_state=126
use_mesa = True

# this is adjusted to use ALL files, not exclude any -- a 70/30 split
df_splits = pd.read_csv('../data/final_shhs1_2_zarrs_with_splits_for_comparison_no_exclusion.csv')

train_shhs_zarrs = df_splits.loc[df_splits.dataset_split!='test', 'file_path'].unique().tolist() # this is a 70/30 split
groups_shhs = [Path(i).stem.split('-')[-1] for i in train_shhs_zarrs]

zarr_files_wsc = glob.glob(os.path.join(data_dir, "wsc_waveforms_no_resampling/*.zarr/"))
groups_wsc = [Path(i).stem.split('-')[-1] for i in zarr_files_wsc]

zarr_files_mros = glob.glob(os.path.join(data_dir, "mros_waveforms_no_resampling/mros-visit1-*.zarr/"))
groups_mros = [Path(i).stem.split('-')[-1] for i in zarr_files_mros]

WSC_MROS_TRAIN_SIZE = 0.7
splitter_wsc = GroupShuffleSplit(n_splits=1, train_size=WSC_MROS_TRAIN_SIZE, random_state=random_state)
splitter_mros = GroupShuffleSplit(n_splits=1, train_size=WSC_MROS_TRAIN_SIZE, random_state=random_state)
train_idxs_wsc, valid_idxs_wsc = next(splitter_wsc.split(X=zarr_files_wsc, groups=groups_wsc))
train_idxs_mros, valid_idxs_mros = next(splitter_mros.split(X=zarr_files_mros, groups=groups_mros))

train_zarrs_wsc = [zarr_files_wsc[i] for i in train_idxs_wsc]
train_zarrs_mros = [zarr_files_mros[i] for i in train_idxs_mros]

all_train_zarrs = train_shhs_zarrs + train_zarrs_wsc + train_zarrs_mros
all_groups = groups_shhs + [Path(i).stem.split('-')[-1] for i in train_zarrs_wsc] + [Path(i).stem.split('-')[-1] for i in train_zarrs_mros]

TRAIN_VAL_SPLIT = 0.9
splitter_train_val = GroupShuffleSplit(n_splits=1, train_size=TRAIN_VAL_SPLIT, random_state=random_state)
train_idxs, valid_idxs = next(splitter_wsc.split(X=all_train_zarrs, groups=all_groups))

train_zarrs = [all_train_zarrs[i] for i in train_idxs]
val_zarrs = [all_train_zarrs[i] for i in valid_idxs]
#assert set([Path(i).stem.split('-')[-1] for i in train_zarrs]) & set([Path(i).stem.split('-')[-1] for i in val_zarrs]) == set(), "There are patients across split across training and validation"

if use_mesa:
    zarr_files_mesa = glob.glob(os.path.join(data_dir, "mesa_waveforms_no_resampling/*.zarr/")) # all mesas have hypnograms
    random.Random(123).shuffle(zarr_files_mesa) # same seed as usleep
    zarr_files_mesa = zarr_files_mesa[100:] # use these for training/validation (first 100 for testing)
    groups_mesa = [Path(i).stem.split('-')[-1] for i in zarr_files_mesa]
    splitter_mesa = GroupShuffleSplit(n_splits=1, train_size=TRAIN_VAL_SPLIT, random_state=random_state)
    train_idxs_mesa, valid_idxs_mesa = next(splitter_mesa.split(X=zarr_files_mesa, groups=groups_mesa))
    train_zarrs_mesa = [zarr_files_mesa[i] for i in train_idxs_mesa]
    val_zarrs_mesa = [zarr_files_mesa[i] for i in valid_idxs_mesa]
    train_zarrs += train_zarrs_mesa
    val_zarrs += val_zarrs_mesa

#channels = ['ECG', 'EOG(L)', 'EMG', 'EEG', 'SaO2', 'THOR RES', 'ABDO RES']
channels = [['ECG', 'ECG (L-R)', 'EKG'],
            ['EOG(L)', 'E1', 'E1-M2', 'EOG-L'],
            ['EMG', 'cchin_l', 'chin', 'EMG (L-R)'],
            ['EEG', 'C3_M2', "C4-M1", "C3-M2", 'EEG3'],
            ['SaO2', 'spo2', 'SpO2'],
            ['THOR RES', 'thorax', 'Thoracic', 'Chest', 'Thor'],
            ['ABDO RES', 'abdomen', 'Abdominal', 'ABD', 'Abdo']]

# channels = [['EEG', 'C3_M2', "C4-M1", "C3-M2", 'EEG3'],
#             ['EOG(L)', 'E1', 'E1-M2', 'EOG-L'],
#             ['EMG', 'cchin_l', 'chin', 'EMG (L-R)']]

#channels = ['EEG', 'EOG(L)', 'EMG']
c_in = len(channels)
frequency = 125
win_length = 750# 3 seconds, it seems like 3600 patches is the maximum amount, maybeee 4320 (5 secs)
#n_fft = win_length
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len_sec = (8*3600) # for dataloader
seq_len_sec = sample_stride = max_seq_len_sec#3*3600 # for dataloader
max_seq_len = seq_len_sec*frequency if seq_len_sec is not None else max_seq_len_sec*frequency # for model

include_partial_samples = True # these should all be true if doing full length sequences

mask_ratio = 0.1

metrics = [cosine_similarity,mse,mae]

use_mask = False

arch = dict(c_in=c_in,
            win_length=win_length,
            hop_length=hop_length,
            max_seq_len=max_seq_len,
            #time_domain=True,
            use_revin=True,
            dim1reduce = False,
            use_flash_attn=False, 
            affine=True, # need to test with both true and false
            augmentations=['jitter_zero_mask'],#jitter_zero_mask', 'reverse_sequence', 'shuffle_channels'],
            mask_ratio=mask_ratio,
            n_layers=3,
            d_model=512,
            n_heads=4,
            shared_embedding=False,
            d_ff=2048,
            norm='BatchNorm',
            attn_dropout=0.,
            dropout=0.1,
            act="gelu", 
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pretrain_head=True,
            pretrain_head_n_layers=1,
            pretrain_head_dropout=0.
            )

filename = f"{model_run}-{name}-{loss_func}-loss-" + "{epoch:02d}-{val_loss:.5f}"
checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=3, monitor="val_loss", mode='min', filename=filename)

wandb_logger = WandbLogger(project=f'{model_run}', offline=False, name=name, save_dir=models_dir)
wandb_logger.log_hyperparams(arch)

# train model
if __name__ == "__main__":
    pl.seed_everything(random_state)

    train_ds = SelfSupervisedTimeFrequencyDataset(zarr_files=train_zarrs,
                                            channels=channels, 
                                            frequency=frequency,
                                            trim_wake_epochs=trim_wake_epochs,
                                            return_hypnogram_every_sec=return_hypnogram_every_sec,
                                            hypnogram_frequency=hypnogram_frequency,
                                            hypnogram_padding_mask=hypnogram_padding_mask,
                                            scale_channels=scale_channels, # we could check tuning on this
                                            start_offset_sec=0,
                                            clip_interpolations=None, # we could tune this
                                            include_partial_samples=include_partial_samples, 
                                            return_sequence_padding_mask=return_sequence_padding_mask,
                                            butterworth_filters=ALL_FREQUENCY_FILTERS, # dictionary of low pass, high pass, and bandpass dictionary to perform on channels
                                            median_filter_kernel_size=3, # if not none, will apply median filter with kernel size
                                            voltage_channels=VOLTAGE_CHANNELS, # if not None, these channels units will be looked at and changed to microvolts from mv uv etc.
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride)

    val_ds = SelfSupervisedTimeFrequencyDataset(zarr_files=val_zarrs, 
                                            channels=channels, 
                                            frequency=frequency,
                                            trim_wake_epochs=trim_wake_epochs,
                                            return_hypnogram_every_sec=return_hypnogram_every_sec,
                                            hypnogram_frequency=hypnogram_frequency,
                                            hypnogram_padding_mask=hypnogram_padding_mask,
                                            scale_channels=scale_channels, 
                                            start_offset_sec=0,
                                            clip_interpolations=None,
                                            include_partial_samples=include_partial_samples, 
                                            return_sequence_padding_mask=return_sequence_padding_mask,
                                            butterworth_filters=ALL_FREQUENCY_FILTERS, # dictionary of low pass, high pass, and bandpass dictionary to perform on channels
                                            median_filter_kernel_size=3, # if not none, will apply median filter with kernel size
                                            voltage_channels=VOLTAGE_CHANNELS, # if not None, these channels units will be looked at and changed to microvolts from mv uv etc.
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride, 
                                            time_channel_scales = None)

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    
    
    patchfreq_model = PatchTFTSimpleLightning(precalculate_onebatch_stft_stats=False, 
                                            use_mask=False, 
                                            learning_rate=learning_rate, 
                                            loss_func=loss_func, 
                                            train_size=len(train_ds), 
                                            max_lr = 0.01, 
                                            metrics=[], 
                                            channels=channels, 
                                            epochs=EPOCHS, 
                                            batch_size=BATCHSIZE, 
                                            use_sequence_padding_mask=use_sequence_padding_mask,
                                            huber_delta=1,
                                            cross_attention=False,
                                            patch_continuity_loss=0,
                                            scheduler_type='onecycle',
                                            optimizer_type='adamw',
                                            weight_decay=1e-4,
                                            **arch)
    
    trainer = pl.Trainer(precision="32",
                     enable_checkpointing=True, # not me
                     enable_progress_bar=True, # not me
                     enable_model_summary=True, # not me
                     logger=wandb_logger, 
                     strategy="ddp",
                     sync_batchnorm=True,
                     val_check_interval=val_check_interval,
                     log_every_n_steps=50,
                     gradient_clip_val=gradient_clip_val,
                     gradient_clip_algorithm=gradient_clip_algorithm,
                     num_sanity_val_steps=0, # speed up
                     detect_anomaly=False, # speed up, though defualt
                     profiler=None, # this is the default, not me
                     accelerator="gpu", 
                     accumulate_grad_batches=accumulate_grad_batches,
                     devices=n_gpus, 
                     default_root_dir=models_dir, 
                     max_epochs=EPOCHS, 
                     fast_dev_run=False,
                     callbacks=[checkpoint_callback])
    
    #tuner = Tuner(trainer)
    # Run learning rate finder
    #lr_finder = tuner.lr_find(patchfreq_model, train_dataloaders=train_loader)

    # Pick point based on plot, or get suggestion
    #new_lr = lr_finder.suggestion()

    # # update hparams of the model
    #patchfreq_model.hparams.learning_rate = new_lr if new_lr is not None else patchfreq_model.hparams.learning_rate
    
    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
    #trainer.save_checkpoint(os.path.join(models_dir, f"{model_run}-last-epoch.ckpt"))
