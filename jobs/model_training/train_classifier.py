import torch, pandas as pd, os, glob, lightning.pytorch as pl, torch.nn as nn

from pftsleep.train import PatchTFTSleepStage, PatchTFTSimpleLightning
from pftsleep.heads import RNNProbingHeadExperimental
from pftsleep.slumber import HypnogramTimeFrequencyDataset, ALL_FREQUENCY_FILTERS, VOLTAGE_CHANNELS

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from pathlib import Path
import random

from pftsleep.augmentations import IntraClassCutMix1d, IntraClassCutMixBatch, MixupCallback

wandb_offline = False

val_check_interval = 0.5
weight_decay = 1e-3

max_lr = 0.01

use_sequence_padding_mask = return_sequence_padding_mask = True
trim_wake_epochs = True # note that class weights are gonna be different with this set to true
#cutmix_callback = IntraClassCutMix1d(mix_prob=0.5, return_y_every_sec=30, frequency=125) # this mixes across batches per patch
cutmix_callback = IntraClassCutMixBatch(mix_prob=0.5, intra_class_only=False, return_y_every_sec=30, frequency=125, return_sequence_padding_mask=return_sequence_padding_mask) # this one mixes across batches and patches
mixup_callback = None #MixupCallback(num_classes=5, mixup_alpha=0.2, return_sequence_padding_mask=return_sequence_padding_mask, ignore_index=-100)

## with mixup, can only use kldivloss
torch.set_float32_matmul_precision('medium')

models_dir = ""
data_dir = ""
model_run = ''

name = ''

pretrained_encoder_path = os.path.join(models_dir, "")
encoder_model = PatchTFTSimpleLightning.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')

precision = '32'
loss_fxn = 'FocalLoss'

BATCHSIZE = 4
accumulate_grad_batches = 8
EPOCHS = 30
n_gpus = -1
num_workers = 8
learning_rate = 1e-5

label_smoothing = 0 # only works for ce loss
gamma = 2.

random_state=126

gradient_clip_val = 1
use_gradient_clipping=False

rnn_norm = None# 'pre' #'pre' 'post'

fine_tune = False

frequency = 125

y_frequency = 1 # this is the frequency of the hypnogram files in the dataset
scale_channels = False

use_wsc_mros = False
use_mesa = False #2051 / 1611
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

if use_wsc_mros:
    all_train_zarrs = train_zarrs_wsc + train_zarrs_mros
    all_groups = [Path(i).stem.split('-')[-1] for i in train_zarrs_wsc] + [Path(i).stem.split('-')[-1] for i in train_zarrs_mros]
else:
    all_train_zarrs = train_shhs_zarrs
    all_groups = groups_shhs

TRAIN_VAL_SPLIT = 0.9
splitter_train_val = GroupShuffleSplit(n_splits=1, train_size=TRAIN_VAL_SPLIT, random_state=random_state)
train_idxs, valid_idxs = next(splitter_wsc.split(X=all_train_zarrs, groups=all_groups))

# only get those with a hypnogam! but same splits as training self supervised model!
train_zarrs = [all_train_zarrs[i] for i in train_idxs if Path(Path(all_train_zarrs[i])/'hypnogram').exists()]
val_zarrs = [all_train_zarrs[i] for i in valid_idxs if Path(Path(all_train_zarrs[i])/'hypnogram').exists()]

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
#MESA_CHANNELS = [["EKG"], ["EOG-L"], ["EMG"], ["EEG3"], ["SpO2"], ["Thor"], ["Abdo"]]
channels = [['ECG', 'ECG (L-R)', "EKG"],
            ['EOG(L)', 'E1', 'E1-M2', "EOG-L"],
            ['EMG', 'cchin_l', 'chin', 'EMG (L-R)'],
            ['EEG', 'C3_M2', "C4-M1", "C3-M2", "EEG3"],
            ['SaO2', 'spo2', 'SpO2'],
            ['THOR RES', 'thorax', 'Thoracic', 'Chest', "Thor"],
            ['ABDO RES', 'abdomen', 'Abdominal', 'ABD', "Abdo"]]
# cd
# channels = [['EEG', 'C3_M2', "C4-M1", "C3-M2", 'EEG3'],
#             ['EOG(L)', 'E1', 'E1-M2', 'EOG-L'],
#             ['EMG', 'cchin_l', 'chin', 'EMG (L-R)']]
c_in = len(channels)

win_length=frequency*6 # 6 seconds patch 
return_y_every_sec = 30 #win_length/frequency
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len_sec = (8*3600) # for dataloader
seq_len_sec = sample_stride = max_seq_len_sec # for dataloader
max_seq_len = seq_len_sec*frequency if seq_len_sec is not None else max_seq_len_sec*frequency # for model

#####
## this is how it was trained
include_partial_samples = True # these should all be true if doing full length sequences
####
#
n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1

conv_kernel_stride_size = None#(1,64)
conv_out_channels = None

y_padding_mask=-100

class_weights = 1 - torch.as_tensor([0.2778, 0.0367, 0.4084, 0.1336, 0.1436]) # shhs df_splits class weights
class_weights = 1 / torch.as_tensor([0.3004, 0.0423, 0.4182, 0.1080, 0.1311]) # shhs, wsc, mros 70.30 class weights

class_weights = class_weights / class_weights.sum() # normalize
class_weights = None


lp_head = dict(c_in=c_in, # GRU head
                input_size=512,
                hidden_size=1024,
                predict_every_n_patches=5, 
                n_classes=5, 
                num_rnn_layers=2,
                contrastive=contrastive,
                rnn_dropout=0.1, # only runs when num_rnn_layers > 1
                module='GRU',
                bidirectional=True,
                affine=True,
                #shared_embedding=False,
                #n_linear_layers=1,
                #act='gelu',
		        pool='average', # majority is an option
                pre_norm=rnn_norm == 'pre', 
                mlp_final_head=True,
                linear_dropout=0.1,
                temperature=2 # only used for majority pooling
            )

lp_model = RNNProbingHeadExperimental(**lp_head)

filename = f"{model_run}-{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=3, monitor="val_loss", mode='min', filename=filename)

add_params = {'encoder_path':pretrained_encoder_path, 'trim_wake_epochs':trim_wake_epochs, 'cutmix_callback':True if cutmix_callback is not None else False, 'mixup_callback':True if mixup_callback is not None else False}

wandb_logger = WandbLogger(project=f"{model_run}", offline=wandb_offline, name=name, save_dir=models_dir)
wandb_logger.log_hyperparams({**dict(encoder_model.hparams), **{f"lp_head_{k}":v for k,v in lp_head.items()}, **add_params})

callbacks = [checkpoint_callback]
if mixup_callback is not None:
    callbacks.append(mixup_callback)
if cutmix_callback is not None:
    callbacks.append(cutmix_callback)
callbacks.reverse()
# train model
if __name__ == "__main__":
    pl.seed_everything(random_state)

    train_ds = HypnogramTimeFrequencyDataset(zarr_files=train_zarrs,
                                             frequency=frequency,
                                             y_frequency=y_frequency,
                                            return_y_every_sec=return_y_every_sec,
                                            trim_wake_epochs=trim_wake_epochs,
                                            include_partial_samples=include_partial_samples,
                                            return_sequence_padding_mask=return_sequence_padding_mask,
                                            y_padding_mask=y_padding_mask,
                                            channels=channels, 
                                            scale_channels=scale_channels, 
                                            butterworth_filters=ALL_FREQUENCY_FILTERS, # dictionary of low pass, high pass, and bandpass dictionary to perform on channels
                                            median_filter_kernel_size=3, # if not none, will apply median filter with kernel size
                                            voltage_channels=VOLTAGE_CHANNELS, # if not None, these channels units will be looked at and changed to microvolts from mv uv etc.
                                            clip_interpolations=None,
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride)
    

    val_ds = HypnogramTimeFrequencyDataset(zarr_files=val_zarrs, 
                                       return_y_every_sec=return_y_every_sec, 
                                       frequency=frequency,
                                       y_frequency=y_frequency,
                                       trim_wake_epochs=trim_wake_epochs,
                                       include_partial_samples=include_partial_samples,
                                       return_sequence_padding_mask=return_sequence_padding_mask,
                                       y_padding_mask=y_padding_mask,
                                       scale_channels=scale_channels, 
                                       channels=channels, 
                                       butterworth_filters=ALL_FREQUENCY_FILTERS, # dictionary of low pass, high pass, and bandpass dictionary to perform on channels
                                        median_filter_kernel_size=3, # if not none, will apply median filter with kernel size
                                        voltage_channels=VOLTAGE_CHANNELS, # if not None, these channels units will be looked at and changed to microvolts from mv uv etc.
                                       clip_interpolations=None,
                                       max_seq_len_sec=max_seq_len_sec, 
                                       sample_seq_len_sec=seq_len_sec, 
                                       sample_stride_sec=sample_stride, 
                                       time_channel_scales=train_ds.time_channel_scales if scale_channels else None)
    
    patchmeup_model = PatchTFTSleepStage(learning_rate=learning_rate, 
                                         class_weights=class_weights,
                                         linear_probing_head=lp_model,
                                         label_smoothing=label_smoothing,
                                         loss_fxn=loss_fxn,
                                         fine_tune=fine_tune,
                                         gamma=gamma,
                                         weight_decay=weight_decay,
                                         use_sequence_padding_mask=use_sequence_padding_mask,
                                         y_padding_mask=y_padding_mask,
                                         pretrained_encoder_path=None,
                                         preloaded_model=encoder_model, 
                                         torch_model_name='model', 
                                         remove_pretrain_layers=['head', 'mask', 'conv_head'],
                                         train_size=len(train_ds), 
                                         scheduler_type='onecycle',
                                         optimizer_type='adamw',
                                         max_lr = max_lr,
                                         metrics=[], 
                                         epochs=EPOCHS, 
                                         batch_size=BATCHSIZE
                                         )
    

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)

    trainer = pl.Trainer(precision=precision,
                     enable_checkpointing=True, # not me
                     enable_progress_bar=True, # not me
                     enable_model_summary=True, # not me
                     logger=wandb_logger,
                     val_check_interval=val_check_interval,
                     sync_batchnorm=True,
                     strategy="ddp",
                     gradient_clip_val=gradient_clip_val,
                     gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                     log_every_n_steps=50,
                     num_sanity_val_steps=0, # speed up
                     detect_anomaly=False, # speed up, though defualt
                     profiler=None, # this is the default, not me
                     accelerator="gpu", 
                     accumulate_grad_batches=accumulate_grad_batches,
                     devices=n_gpus, 
                     default_root_dir=models_dir, 
                     max_epochs=EPOCHS, 
                     fast_dev_run=False,
                     callbacks=callbacks)
    
#    tuner = Tuner(trainer)
#     # # Run learning rate finder
#    lr_finder = tuner.lr_find(patchmeup_model, train_dataloaders=train_loader)

#     # # Pick point based on plot, or get suggestion
#    new_lr = lr_finder.suggestion()

#     # # # update hparams of the model
#    patchmeup_model.hparams.learning_rate = new_lr if new_lr is not None else patchmeup_model.hparams.learning_rate
    
    trainer.fit(model=patchmeup_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
    #trainer.save_checkpoint(os.path.join(models_dir, f"{model_run}-last-epoch.ckpt"))
