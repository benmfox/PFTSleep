"""Potentially helpful layers for your models"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_layers.ipynb.

# %% auto 0
__all__ = ['PatchEncoder', 'PositionalEncoding', 'tAPE', 'Mask', 'PatchAugmentations', 'EmbeddingAugmentations', 'Patch', 'STFT',
           'FFT', 'RevIN', 'MultiheadFlashAttention', 'ScaledDotProductAttention', 'MultiheadAttentionCustom',
           'Attention_Rel_Scl', 'MaskedAutogressionFeedForward', 'MaskedAutogressionFeedForward2', 'Identity',
           'Transpose', 'get_activation_fn']

# %% ../nbs/10_layers.ipynb 3
import torch, numpy as np, torch.nn.functional as F

from typing import Optional
from torch import nn
from torch import Tensor
from torch.fft import fft
from .augmentations import create_patch, mask_patches_simple, jitter_augmentation, shuffle_dim, reverse_sequence
from .signal import stft
from torch.nn.attention import SDPBackend, sdpa_kernel

# %% ../nbs/10_layers.ipynb 5
class PatchEncoder(nn.Module):
    def __init__(self, 
                 c_in, # the number of input channels
                 patch_len, # the length of the patches (either stft or interval length)
                 d_model, # the dimension of the initial linear layers for inputting patches into transformer
                 shared_embedding, # indicator of whether to project each channel individually or together
                 ):
        super().__init__()

        self.shared_embedding = shared_embedding
        self.n_vars = c_in

        # Input encoding: projection of feature vectors onto a d-dim vector space
        ## note that this could be an MLP too, if you want
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

    def forward(self, x) -> Tensor:          
        """
        input: x: tensor [bs x num_patch x nvars x patch_len]
        returns: x: tensor [bs x num_patch x nvars x d_model]
        """
        # Input embedding
        if not self.shared_embedding:
            x_out = []
            for i in range(self.n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x) # x: [bs x num_patch x nvars x d_model]
        return x

# %% ../nbs/10_layers.ipynb 7
class PositionalEncoding(nn.Module):
    def __init__(self, 
                 num_patch, # number of patches of time series or stft in input
                 d_model, # dimension of patch embeddings
                 #dropout=0.1 # dropout value
                 ):
        super().__init__()
        self.num_patch = num_patch
        self.d_model = d_model
        
        # Positional encoding - learned
        self.W_pos =  nn.Parameter(torch.empty((num_patch, d_model)))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        input: x: [bs * nvars x num_patch x d_model]
        returns: x: [bs * nvars x num_patch x d_model]
        """
        x = x + self.W_pos
        return x


# %% ../nbs/10_layers.ipynb 8
class tAPE(nn.Module):
    """
    time Absolute Position Encoding
    Adapted from tsai
    """

    def __init__(self, 
        d_model:int, # the embedding dimension
        seq_len:int, # the max. length of the incoming sequence or num patches
        #dropout:float=0., # dropout value
        scale_factor=1.0
        ):
        super().__init__()
        
        pe = torch.zeros(seq_len, d_model)  # positional encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/seq_len)) # this is the difference between normal PE and tAPE, scaling (d_model/seq_len)
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/seq_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): # [batch size, sequence length, embed dim]
        x = x + self.pe
        return x

# %% ../nbs/10_layers.ipynb 10
class Mask(nn.Module):
    def __init__(self, mask_type, mask_ratio, return_mask=True):
        super().__init__()
        assert mask_type in ['patch', 'jitter_zero']
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.return_mask = return_mask
    
    def forward(self, x):
        if self.mask_type == 'jitter_zero':
            # mask is a number with the number of masks applied in this function
            x_masked, mask = jitter_augmentation(x, mask_ratio=(self.mask_ratio/2), jitter_ratio=(self.mask_ratio/2))
        else:
            # padding mask is currently not implemented here.. tbh not sure if its needed
            x_masked, mask = mask_patches_simple(x, mask_ratio=self.mask_ratio)
        if self.return_mask:
            return x_masked, mask
        else:
            return x_masked


# %% ../nbs/10_layers.ipynb 13
class PatchAugmentations(nn.Module):
    def __init__(self, augmentations=['patch_mask', 'jitter_zero_mask', 'reverse_sequence', 'shuffle_channels'], patch_mask_ratio=0., jitter_zero_mask_ratio=0.):
        super().__init__()
        if 'patch_mask' in augmentations:
            assert patch_mask_ratio >= 0.
        if 'jitter_zero_mask' in augmentations:
            assert jitter_zero_mask_ratio >= 0.
        self.augmentations = augmentations
        self.patch_mask_ratio = patch_mask_ratio
        self.jitter_zero_mask_ratio = jitter_zero_mask_ratio
    def forward(self, x):
        if 'patch_mask' in self.augmentations and 'jitter_zero_mask' in self.augmentations:
            mask_choice = np.random.choice(['patch_mask', 'jitter_zero_mask'], replace=True)
        else:
            mask_choice = None
        augs = self.augmentations.copy()
        np.random.shuffle(augs)
        for augmentation in augs:
            if augmentation == 'jitter_zero_mask' and (mask_choice == 'jitter_zero_mask' or mask_choice is None):
                # mask is a number with the number of masks applied in this function
                x, _ = jitter_augmentation(x, mask_ratio=(self.jitter_zero_mask_ratio/2), jitter_ratio=(self.jitter_zero_mask_ratio/2))
            if augmentation == 'patch_mask' and (mask_choice == 'patch_mask' or mask_choice is None):
                # padding mask is currently not implemented here.. tbh not sure if its needed
                x, _ = mask_patches_simple(x, mask_ratio=self.patch_mask_ratio)
            if augmentation == 'shuffle_channels':
                x = shuffle_dim(x, dim=2, p=0.5)
            if augmentation == 'reverse_sequence':
                x = reverse_sequence(x, seq_dim=(-1,), p=0.5)
        return x

# %% ../nbs/10_layers.ipynb 15
class EmbeddingAugmentations(nn.Module):
    def __init__(self, augmentations=['shuffle_dims', 'jitter_zero_mask', 'patch_mask'], dims_to_shuffle = [1,2,3], patch_mask_ratio=0., jitter_zero_mask_ratio=0.):
        super().__init__()
        assert set(augmentations) - set(['shuffle_dims', 'jitter_zero_mask', 'patch_mask']) == set(), f"Augmentations must be in {set(['shuffle_dims', 'jitter_zero_mask', 'patch_mask'])}."
        if 'patch_mask' in augmentations:
            assert patch_mask_ratio >= 0.
        if 'jitter_zero_mask' in augmentations:
            assert jitter_zero_mask_ratio >= 0.
        self.augmentations = augmentations
        self.patch_mask_ratio = patch_mask_ratio
        self.jitter_zero_mask_ratio = jitter_zero_mask_ratio
        self.dims_to_shuffle = dims_to_shuffle
    
    def forward(self, x):
        """
        Input is an embedding: 
        x: [bs x n channels x d model x n patches]
        dims correspond to 1 = channels 2 = n patches 3 = d model
        returns: [bs x n channels x d model x n patches]
        """
        x = x.permute(0,1,3,2)
        if 'patch_mask' in self.augmentations and 'jitter_zero_mask' in self.augmentations:
            mask_choice = np.random.choice(['patch_mask', 'jitter_zero_mask'], replace=True)
        else:
            mask_choice = None
        augs = self.augmentations
        np.random.shuffle(augs)
        for augmentation in augs:
            if augmentation == 'jitter_zero_mask' and (mask_choice == 'jitter_zero_mask' or mask_choice is None):
                # mask is a number with the number of masks applied in this function
                x, _ = jitter_augmentation(x, mask_ratio=(self.jitter_zero_mask_ratio/2), jitter_ratio=(self.jitter_zero_mask_ratio/2))
            if augmentation == 'patch_mask' and (mask_choice == 'patch_mask' or mask_choice is None):
                # padding mask is currently not implemented here.. tbh not sure if its needed
                x, _ = mask_patches_simple(x, mask_ratio=self.patch_mask_ratio)
            if augmentation == 'shuffle_dims':
                # shuffle channels, patches, features
                dim_shuffle_order = self.dims_to_shuffle
                np.random.shuffle(dim_shuffle_order)
                for dim in dim_shuffle_order:
                    x = shuffle_dim(x, dim=dim, p=0.5)
        x = x.permute(0,1,3,2)
        return x

# %% ../nbs/10_layers.ipynb 18
class Patch(nn.Module):
    def __init__(self, patch_len, stride, max_seq_len=None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.max_seq_len = max_seq_len
    
    def forward(self, x, constant_pad=False, constant_pad_value=0):
        x = create_patch(x, patch_len=self.patch_len, stride=self.stride, constant_pad=constant_pad, constant_pad_value=constant_pad_value, max_seq_len=self.max_seq_len)
        return x

# %% ../nbs/10_layers.ipynb 19
class STFT(nn.Module):
    def __init__(self, 
                 n_fft, 
                 win_length,
                 hop_length, 
                 stft_norm, 
                 decibel_scale, 
                 channel_stft_means=None, 
                 channel_stft_stds=None, 
                 pad_win_length_to_nfft=True, 
                 pad_mode='reflect', 
                 center=False, 
                 return_complex=True
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.pad_win_length_to_nfft = pad_win_length_to_nfft
        self.center = center
        self.return_complex = return_complex
        self.stft_norm = stft_norm
        self.decibel_scale = decibel_scale
        self.channel_stft_means = channel_stft_means
        self.channel_stft_stds = channel_stft_stds

    def forward(self, x):
        """
        x: [bs x n_vars x max_seq_len]
        out: [bs x n_vars x n_fft // 2 + 1 x stft_len]
        """
        x_fft = stft(x, 
                     n_fft=self.n_fft, 
                     win_length=self.win_length,
                     normalized=self.stft_norm,
                     pad_win_length_to_nfft=self.pad_win_length_to_nfft,
                     decibel_scale=self.decibel_scale, 
                     channel_stft_means=self.channel_stft_means, 
                     channel_stft_stds=self.channel_stft_stds, 
                     hop_length=self.hop_length, 
                     pad_mode=self.pad_mode, 
                     center=self.center, 
                     return_complex=self.return_complex
                     )
        return x_fft


# %% ../nbs/10_layers.ipynb 20
class FFT(nn.Module):
    def __init__(self, 
                 dim=-1, # dimension to calculate fft over
                 norm='backward' #  "forward" - normalize by 1/n, "backward" - no normalization, "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
                 ):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x):
        """
        This is equivalent to an stft with onesided set to False, hop length of 1, and win_length == n_fft.

        x: [bs x num_patch x n_vars x patch_len]
        out: [bs x num_patch x n_vars x patch_len(freq domain)]
        """
        x_fft = fft(x, dim=self.dim, norm=self.norm).abs()
        return x_fft


# %% ../nbs/10_layers.ipynb 22
class RevIN(nn.Module):
    def __init__(self, 
                 num_features: int, # the number of channels or features in the input
                 eps=1e-5, # added to avoid division by zero errors
                 dim_to_reduce=-1, # the dimension to reduce, 
                 affine=True # learning affine parameters bias and weight per channel
                 ):
        """
        
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.dim_to_reduce = dim_to_reduce

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1,num_features,1))
            self.affine_bias = nn.Parameter(torch.zeros(1,num_features,1))

    def forward(self, x, mode:bool):
        """
        x: [bs x n_vars x max_seq_len]
        """
        if mode:
            return self._normalize(x)
        else:
            return self._denormalize(x)

    def _normalize(self, x):      
        self.mean = torch.mean(x, dim=self.dim_to_reduce, keepdim=True).detach()
        self.stdev = torch.std(x, dim=self.dim_to_reduce, keepdim=True, unbiased=False).detach() + self.eps

        x = x.sub(self.mean)
        x = x.div(self.stdev)
        
        if self.affine:
            x = x.mul(self.affine_weight)
            x = x.add(self.affine_bias)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x.sub(self.affine_bias)
            x = x.div(self.affine_weight)# + self.eps*self.eps)
        x = x.mul(self.stdev)
        x = x.add(self.mean)
        return x

# %% ../nbs/10_layers.ipynb 25
class MultiheadFlashAttention(nn.Module):
    """Multihead attention layer with optional causal masking.
    Uses flash attention when available in PyTorch 2.0+.
    
    Args:
        n_heads (int): Number of attention heads
        d_model (int): Embedding dimension
        qkv_bias (bool, optional): Use bias in linear layers. Defaults to False.
        is_causal (bool, optional): Use causal masking. Defaults to False.
        attn_dropout (float, optional): Attention dropout probability. Defaults to 0.0.
        proj_dropout (float, optional): Dropout probability. Defaults to 0.0.

    Note that when passing in a key paddings mask, it should be a boolean tensor of shape [bs x seq_len]
    where a True value indicates that the key at that position should be ignored for the purposes of attention.
    This is contrary to what the PyTorch documentation suggests, but is correct in this module.
    """, 
    def __init__(self, d_model: int, n_heads: int, qkv_bias: bool=True, 
                 is_causal: bool=False, attn_dropout: float=0.0, proj_dropout: float=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        # # key, query, value projections for all heads, but in a batch, Combined Q,K,V projections
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        head_dim = d_model // n_heads
        self.scale = head_dim ** -0.5
        
        # Regularization
        self.attn_dropout = attn_dropout
        self.resid_dropout = nn.Dropout(proj_dropout)
        
        # Architecture
        self.num_heads = n_heads
        self.embed_dimension = d_model
        self.is_causal = is_causal

    def forward(self, x: Tensor, 
                key_padding_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        x: [bs x seq_len x d_model]
        key_padding_mask: [bs x seq_len]
        """
        batch_size = x.size(0)
        
        # Project to Q,K,V
        qkv = self.c_attn(x)
        embed_dim = qkv.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        # Split and reshape
        query, key, value = qkv.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # Set dropout and causality based on training mode
        attn_dropout = self.attn_dropout if self.training else 0.0
        is_causal = self.is_causal if self.training else False
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [bs x 1 x 1 x seq_len]
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = key_padding_mask.float().masked_fill(key_padding_mask, float('-inf'))
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, query.size(-2), -1)
            if key_padding_mask.stride(-1) != 1:
                # see https://github.com/pytorch/pytorch/issues/127523
                key_padding_mask = torch.empty_like(key_padding_mask, memory_format=torch.contiguous_format).copy_(key_padding_mask)

        # Attention (with flash attention when available)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=key_padding_mask,  # PyTorch handles mask preprocessing internally, can pass attn_mask or key_padding_mask
                dropout_p=attn_dropout, 
                is_causal=is_causal,
                scale=self.scale
            )
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * head_dim)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

# %% ../nbs/10_layers.ipynb 27
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        ## this is the b ottleneck in your code
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        ## if implementing attention or key padding mask - https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580
        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:# and torch.any(key_padding_mask): - not sure if this is needed                         # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), -np.inf)

        # normalize the attention weights
        # this is another bottleneck (softmax)
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(attn_mask, 0.0)
       
        if key_padding_mask is not None:# and torch.any(key_padding_mask):
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0.0)
       
        attn_weights = self.attn_dropout(attn_weights)
        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


# %% ../nbs/10_layers.ipynb 28
class MultiheadAttentionCustom(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# %% ../nbs/10_layers.ipynb 32
class Attention_Rel_Scl(nn.Module):
    def __init__(self, 
        d_model:int, # Embedding dimension
        n_heads:int, # number of attention heads
        seq_len:int, # sequence length or num patches
        d_k:int=None, # key dimension
        d_v:int=None, # value dimension
        res_attention:bool=False, # whether to use residual attention
        attn_dropout:float=0., # dropout for attention
        lsa:bool=False, # whether to use LSA, trainable paramater for scaling
        proj_dropout:float=0., # dropout for projection
        qkv_bias:bool=True, # bias for q, k, v
        ):
        """
        Adapted from tsai
        Added residual attention and dropout
        """
        super().__init__()

        self.seq_len = seq_len
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.query = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.key = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.value = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), n_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)), indexing="xy")
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, x, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        batch_size = x.shape[0]
        
        q = self.query(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.value(x).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_scores = torch.matmul(q, k) * self.scale # [seq_len, seq_len]
        if prev is not None: attn_scores = attn_scores + prev

        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:# and torch.any(key_padding_mask): - not sure if this is needed                         # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), -np.inf)
        
        attn_weights = F.softmax(attn_scores, dim=-1)

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.n_heads))
        relative_bias = relative_bias.reshape(self.seq_len, self.seq_len, -1).permute(2, 0, 1).unsqueeze(0)
        attn_weights = attn_weights + relative_bias

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(attn_mask, 0.0)
       
        if key_padding_mask is not None:# and torch.any(key_padding_mask):
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0.0)

        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v) # [batch_size, n_heads, seq_len, d_head]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        out = self.to_out(out)
        if self.res_attention: 
            return out, attn_weights, attn_scores
        else: 
            return out, attn_weights

# %% ../nbs/10_layers.ipynb 36
class MaskedAutogressionFeedForward(nn.Module):
    def __init__(self, 
                 c_in, # the number of input channels
                 patch_len, # the length of the patches (either stft or interval length)
                 d_model, # the dimension of the initial linear layers for inputting patches into transformer
                 shared_recreation=True, # indicator of whether to project each channel individually or together
                 ):
        super().__init__()

        self.shared_recreation = shared_recreation
        self.n_vars = c_in

        # Input encoding: projection of feature vectors onto a d-dim vector space
        ## note that this could be an MLP too, if you want
        if not shared_recreation:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(d_model, patch_len))
        else:
            self.W_P = nn.Linear(d_model, patch_len)

    def forward(self, x) -> Tensor:          
        """
        input: x: tensor [bs x nvars x d_model x num_patch]
        returns: x: tensor [bs x num_patch x nvars x d_model]
        """
        # Input embedding
        x = x.permute(0,3,1,2) # [bs x num_patch x nvars x d_model]
        if not self.shared_recreation:
            x_out = []
            for i in range(self.n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x) # x: [bs x num_patch x nvars x patch_len]
        return x
    
class MaskedAutogressionFeedForward2(nn.Module):
    def __init__(self, d_model, patch_len, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers 
        self.layers = []
        n_nuerons = int((patch_len - d_model)/self.n_layers)
        if self.n_layers == 1:
            self.layers.append(nn.Sequential(nn.Linear(d_model, patch_len), nn.Dropout(dropout)))
        elif self.n_layers == 0:
            # do nothing
            self.layers.append(nn.Identity())
        else:
            for i in range(self.n_layers-1):
                self.layers.append(nn.Sequential(nn.Linear(n_nuerons*i+d_model, n_nuerons*(i+1)+d_model), nn.ReLU(), nn.Dropout(dropout)))
            self.layers.append(nn.Sequential(nn.Linear(n_nuerons*(i+1)+d_model, patch_len)))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.transpose(2,3)
        x = self.layers(x)
        x = x.permute(0,2,1,3)
        return x

# %% ../nbs/10_layers.ipynb 38
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return x

# %% ../nbs/10_layers.ipynb 39
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')
