"""A good use of time, no doubt."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_transformers.ipynb.

# %% auto 0
__all__ = ['TSTEncoderLayer', 'PatchTFTSimple']

# %% ../nbs/04_transformers.ipynb 3
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import warnings
from .layers import *

# %% ../nbs/04_transformers.ipynb 5
class TSTEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, # dimension of patch embeddings
                 n_heads, # number of attention heads per layer
                 d_ff=256, # dimension of feedforward layer in each transformer layer
                 store_attn=False, # indicator of whether or not to store attention
                 norm='BatchNorm',
                 relative_attn_type='vanilla', # options include vaniall or eRPE
                 use_flash_attn=False, # indicator to use flash attention
                 num_patches=None, # num patches required for eRPE attn
                 attn_dropout=0, 
                 dropout=0., 
                 bias=True, 
                 activation="gelu", 
                 res_attention=False, 
                 pre_norm=False
                ):
        super().__init__()
        
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.use_flash_attn = use_flash_attn
        self.store_attn = store_attn
        if self.use_flash_attn and self.store_attn:
            warnings.warn("Flash attention does not support storing attention, setting store_attn to False")
            self.store_attn = False
        if relative_attn_type == 'eRPE':
            assert num_patches is not None, "You must provide a num_patches for eRPE"
            self.self_attn = Attention_Rel_Scl(d_model, n_heads=n_heads, seq_len=num_patches, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=dropout)
        else:
            self.self_attn = MultiheadAttentionCustom(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        if use_flash_attn:
            self.self_attn = MultiheadFlashAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout) 
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias), 
                                get_activation_fn(activation), # note do not put functions in sequential, it makes things non-deterministic
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, prev=prev, key_padding_mask=key_padding_mask)
        elif not self.use_flash_attn:
            src2, attn = self.self_attn(src, key_padding_mask=key_padding_mask)
        else:
            src2 = self.self_attn(src, key_padding_mask=key_padding_mask)
        
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward

        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


# %% ../nbs/04_transformers.ipynb 7
class PatchTFTSimple(nn.Module):
     def __init__(self,
                  c_in:int, # the number of input channels
                  win_length, # the length of the patch of time/interval or short time ft windown length (when time_domain=False)
                  hop_length, # the length of the distance between each patch/fft
                  max_seq_len, # maximum sequence len
                  time_domain=True,
                  pos_encoding_type='learned', # options include learned or tAPE
                  relative_attn_type='vanilla', # options include vanilla or eRPE
                  use_flash_attn=False, # indicator to use flash attention
                  use_revin=True, # if time_domain is true, whether or not to instance normalize time data
                  dim1reduce=False, # indicator to normalize by timepoint in revin
                  affine=True, # if time_domain is true, whether or not to learn revin normalization parameters 
                  mask_ratio=0.1, # amount of signal to mask
                  augmentations=['patch_mask', 'jitter_zero_mask', 'reverse_sequence', 'shuffle_channels'], # the type of mask to use, options are patch or jitter_zero
                  n_layers:int=2, # the number of transformer encoder layers to use
                  d_model=512, # the dimension of the input to the transofmrer encoder
                  n_heads=2, # the number of heads in each layer
                  shared_embedding=False, # indicator for whether or not each channel should be projected with its own set of linear weights to the encoder dimension
                  d_ff:int=2048, # the feedforward layer size in the transformer
                  norm:str='BatchNorm', # BatchNorm or LayerNorm during trianing
                  attn_dropout:float=0., # dropout in attention
                  dropout:float=0.1, # dropout for linear layers
                  act:str="gelu", # activation function
                  res_attention:bool=True, # whether to use residual attention
                  pre_norm:bool=False, # indicator to pre batch or layer norm 
                  store_attn:bool=False, # indicator to store attention
                  pretrain_head=True, # indicator to include a pretraining head
                  pretrain_head_n_layers=1, # how many linear layers on the pretrained head
                  pretrain_head_dropout=0., # dropout applied to pretrain head
                  ):
          super().__init__()
          self.c_in = c_in # original c_in without convolution
          self.shared_embedding = shared_embedding
          self.d_model = d_model # original d_model
          self.use_revin = use_revin
          self.affine = affine
          self.time_domain = time_domain
          self.pretrain_head = pretrain_head
          self.use_flash_attn = use_flash_attn
          if use_flash_attn and res_attention:
               warnings.warn("Flash attention is not yet implemented for residual attention, setting res_attention=False")
               res_attention = False
          # Instance Normalization (full sequence)
          if self.use_revin:
               self.revin = RevIN(num_features=self.c_in, affine=self.affine, dim_to_reduce=1 if dim1reduce else -1)
          self.num_patch = int((max(max_seq_len, win_length)-win_length) // hop_length + 1)
          if ((max_seq_len-win_length) % hop_length != 0):
               # add one for padding if above is true, see create_patch fxn for more details
               self.num_patch += 1
          self.patch_len = win_length
          self.patch_layer = Patch(patch_len=win_length, stride=hop_length, max_seq_len=max_seq_len)
          if not self.time_domain:
               self.fft = FFT(dim=-1)
          
          # Patch Embedding
          self.patch_encoder = PatchEncoder(c_in=self.c_in, patch_len=self.patch_len, d_model = self.d_model, shared_embedding=shared_embedding)
          # Positional Encoding
          if pos_encoding_type.lower() == 'tape':
               self.pe = tAPE(d_model=self.d_model, seq_len=self.num_patch, scale_factor=1.0)
          else:
               self.pe = PositionalEncoding(num_patch=self.num_patch, d_model=self.d_model)
          # residual dropout
          self.dropout = nn.Dropout(dropout)
          # time series transformer layers/Encoder
          self.layers = nn.ModuleList([TSTEncoderLayer(d_model=self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                       attn_dropout=attn_dropout, dropout=dropout, relative_attn_type=relative_attn_type, num_patches=self.num_patch,
                                                       activation=act, res_attention=res_attention, use_flash_attn=use_flash_attn,
                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
          self.res_attention = res_attention

          # Head
          self.mask = PatchAugmentations(augmentations=augmentations, patch_mask_ratio=mask_ratio, jitter_zero_mask_ratio=mask_ratio)
          if self.pretrain_head:
               self.head = MaskedAutogressionFeedForward(c_in = self.c_in, patch_len = self.patch_len, d_model = self.d_model, shared_recreation=self.shared_embedding)
         
     def forward(self, z, sequence_padding_mask=None):
          """
          input from ds is [bs x n_vars x max_seq_len]
          z: tensor [bs x num_patch x n_vars x patch_len]
          """
          bs = z.shape[0]
          # REVIN
          if self.use_revin:
               z = self.revin(z, mode=True) # z: [bs x n_vars x max_seq_len] dont passs sequence pad mask to revin if dim=(1,) - it doesnt matter

          z = self.patch_layer(z, constant_pad=True, constant_pad_value=0) # z: [bs x num_patch x n_vars x patch_len] pad with 0, same as input padded values
          if not self.time_domain:
               z = self.fft(z)
          Y_true = z.clone().detach()
          # patching for pad mask
          if sequence_padding_mask is not None:
               # create a patched version of the mask for attention and loss masking
               ## pad it as well, with 1 (which is the same padding as the padded input values)
               if self.time_domain:
                    patch_padding_mask = self.patch_layer(sequence_padding_mask, constant_pad=True, constant_pad_value=1).detach() # patch_padding_mask: [bs x num_patch x 1 x patch_len]
                    key_padding_mask = torch.all(patch_padding_mask, -1).squeeze(-1)  # key_padding_mask: [bs x num_patch] calculate mask over patch len for each patch
                    key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, self.c_in, -1) # key_padding_mask: [bs x n_vars x num_patch]
               else:
                    # fft will be 0 for zero values / values with no frequency
                    key_padding_mask = torch.all(z, -1) # calculate mask over patch len for each patch
                    key_padding_mask = (key_padding_mask[:,:,:1]==0).unsqueeze(1).expand(-1, self.c_in, -1) # key padding mask is true for patches with non-zero,, only need one channel dimension since channels same

               key_padding_mask = torch.reshape(key_padding_mask, (-1,self.num_patch)) # key_padding_mask: [bs * nvars x num_patch]
          else:
               patch_padding_mask,key_padding_mask = None, None
          # MASKING
          if self.training and self.pretrain_head:
               z = self.mask(z) # z: [bs x num_patch x n_vars x patch_len] not implementing padding mask for masking, it shouldnt matter expects: [bs x num_patch x n_vars x patch_len]

          # EMBEDDING
          z = self.patch_encoder(z) # z: [bs x num_patch x nvars x d_model]
          z = z.transpose(1,2) # z: [bs x nvars x num_patch x d_model]
          # positional encoding
          z = torch.reshape(z, (bs * self.c_in, self.num_patch, self.d_model)) # u: [bs * nvars x num_patch x d_model]

          z = self.pe(z) # z: [bs * nvars x num_patch x d_model]

          # residual dropout
          z = self.dropout(z) # z: [bs * nvars x num_patch x d_model] 
          
          # encoder layers
          if self.res_attention:
               scores = None
               for mod in self.layers:
                    z, scores = mod(z, prev=scores, key_padding_mask=key_padding_mask) # z: [bs * n_vars x num_patch x d_model], scores: [bs * n_vars x n_heads x num_patch x num_patch]
          else:
               for mod in self.layers: 
                    z = mod(z, key_padding_mask=key_padding_mask) # z: [bs * n_vars x num_patch x d_model]
          if key_padding_mask is not None:
               key_padding_mask = key_padding_mask.reshape(-1, self.c_in, self.num_patch)[:,0,:] # key_padding_mask: [bs x num_patch]
          z = torch.reshape(z, (-1, self.c_in, self.num_patch, self.d_model)) # z: [bs x nvars x num_patch x d_model]
          z = z.permute(0,1,3,2) # z: [bs x nvars x d_model x num_patch]
          if self.pretrain_head:
               Y_pred = self.head(z)
          mask=None
          # PRETRAIN HEAD
          if self.training and self.pretrain_head:
               return Y_pred, Y_true, mask, key_padding_mask
          elif not self.training and self.pretrain_head:
               return z, Y_pred, Y_true, mask, key_padding_mask # z: [bs x nvars x d_model x num_patch]
          else:
               return z