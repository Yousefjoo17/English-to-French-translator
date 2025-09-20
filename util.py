#tensors shapes guides : https://chatgpt.com/share/68b97ed7-2f90-800f-b8bc-f8a6d809726a

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
# subsequent_mask function generates a mask specifically for a sequence, instructing the model to focus solely on the actual sequence
# and disregard the padded zeros at the end, which are used only to standardize sequence lengths. 
def subsequent_mask(size): 
    attn_shape = (1, size, size) #(1, seq_len-1, seq_len-1)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # Builds an upper triangular matrix (above the diagonal is 1 otherise is 0) and Converts matrix to 8-bit integers.
    output = torch.from_numpy(subsequent_mask) == 0 # replace 0 with True otherwise false: (above the diagonal is False otherise is True)
    return output # (1, seq_len-1, seq_len-1)

# standard mask function, on the other hand, constructs a standard mask for the target sequence. This standard mask
# has the dual role of concealing both the padded zeros and the future tokens in the target sequence. 
def make_std_mask(tgt, pad):
    tgt_mask=(tgt != pad).unsqueeze(-2) # Boolean mask (True where token ≠ padding), and then Adds a dimension at index -2 [batch, 1, seq-1].
    output=tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return output # (128, seq_len-1, seq_len-1)

# define the Batch class
# src: sequence of indexes for the source language; e.g., those for the English phrase "How are you?"
# trg: sequence of indexes for the target language; e.g., those for the French phrase "Comment etes-vous?"
class Batch:
    def __init__(self, src, trg=None, pad=0): 
        src = torch.from_numpy(src).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # Creates a source mask to conceal padding at the end of the sentence :[batch, 1, seq_len]
        if trg is not None:
            trg = torch.from_numpy(trg).to(DEVICE).long()
            self.trg = trg[:, :-1] # creates input to the decoder: it drops the last token
            self.trg_y = trg[:, 1:] # creates output of the decoder: drops the first : drops the first token
            self.trg_mask = make_std_mask(self.trg, pad) # creates a target mask to conceal both padding and future tokens (128, seq_len-1, seq_len-1)
            self.ntokens = (self.trg_y != pad).data.sum()


from torch import nn
# An encoder-decoder transformer
# source language is Englihs which is encoder input. While, target language is French which is initial input of the decoder which
# also needs encoding output
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask) # Source language is encoded into abstract vector representations.
        output = self.decode(memory, src_mask, tgt, tgt_mask) # The decoder uses these vector representations to generate translation in the target language.
        return output #(batch_size, tgt_seq_len-1, d_model)

# Create an encoder
# The Encoder() class takes input x (for example, a batch of English phrases) and the mask (to mask out sequence padding)
# to generate output (vector representations that capture the meanings of the English phrases).  
from copy import deepcopy
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for i in range(N)])
        self.norm = LayerNorm(layer.size) # 256

    def forward(self, x, mask): # x is (128,seq_len,256)
        for layer in self.layers:
            x = layer(x, mask)
            output = self.norm(x)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout): # self_att is an instance of multiheadattention class, and feed_forward and so on
        super().__init__()
        self.self_attn = self_attn # MultiHeadedAttention
        self.feed_forward = feed_forward # PositionwiseFeedForward
        self.sublayer = nn.ModuleList([deepcopy(SublayerConnection(size, dropout)) for i in range(2)])
        self.size = size  # 256

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #The first sublayer in each encoder layer is a multihead self-attention network
        output = self.sublayer[1](x, self.feed_forward) # The second sublayer in each encoder layer is a feed-forward network.
        return output 

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout): # size here is d_model = 256
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        output = x + self.dropout(sublayer(self.norm(x))) # Each sublayer goes through residual connection and layer normalization.
        return output  
# If there were no residual connections:
    # The network might transform the representation so heavily that positional information could get lost after the first few layers.
    # Since positional encoding is only injected at the very start, if it's lost once, it never comes back.

class LayerNorm(nn.Module): #Layer normalization is somewhat similar to batch normalization. 
    def __init__(self, features, eps=1e-6): #features: the number of dimensions per token d_model 256 here
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) # initialized as ones, same size as features. This is the scale parameter (γ in normalization formulas).
        self.b_2 = nn.Parameter(torch.zeros(features)) # initialized as zeros, same size as features.→ This is the bias/shift parameter (β). (nn.Parameters → meaning they are trainable by backpropagation.)
        self.eps = eps

    def forward(self, x): # Normalizes each token feature-wise:
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        output = self.a_2*x_zscore+self.b_2
        return output

# Create a decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for i in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        output = self.norm(x)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([deepcopy(SublayerConnection(size, dropout)) for i in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask): # x here is target language (french) embeddings
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # The first sublayer is a masked multihead self-attention layer. Produces contextualized embeddings for target tokens. the values now encode dependencies between previous target tokens.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask)) # The second sublayer is a cross-attention layer between the target language and the source language. (memory is the encoder output)
        output = self.sublayer[2](x, self.feed_forward) # The third sublayer is a feed-forward network.
        return output # (batch_size, tgt_seq_len−1, d_model)
# source language is Englihs and target is french

# create the model
def create_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout=0.1):
    attn=MultiHeadedAttention(h, d_model).to(DEVICE)
    ff=PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    pos=PositionalEncoding(d_model, dropout).to(DEVICE)
    model = Transformer(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), deepcopy(pos)), # Creates src_embed by passing source language through word embedding and positional encoding
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), deepcopy(pos)), # Creates tgt_embed by passing target language through word embedding and positional encoding
        Generator(d_model, tgt_vocab)).to(DEVICE)
    for p in model.parameters(): #iterate over all learnable parameters of the model (weights and biases in all layers).
        if p.dim() > 1: # Apply the initialization only to multi-dimensional parameters (like weight matrices), but skip 1D ones (like biases).
            nn.init.xavier_uniform_(p) #apply Xavier initialization, which initializes the weights with values drawn from a uniform distribution, but scaled so that the variance of activations stays stable across layers.
    return model.to(DEVICE)
# deepcopy creates a new copy of the module with its own parameters, not tied to the original one.
# withou deepcopy they would share the same weights 
# the deepcopy is needed because Python passes objects by reference, and without it, all layers would unintentionally share the same module.(weights)


# Embeddings are random at the start, but they get trained to capture semantic meaning.
# Embeddings don’t have their own loss function.Instead, they are trained indirectly through the
# loss function of the main task (e.g., language modeling, classification, translation, etc.). So, it's task-dependent not fixed.
import math
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.lut(x) * math.sqrt(self.d_model) # . This multiplication is intended to counterbalance the division by the square root of d_model that occurs later during the computation of attention scores.
        return out

# in the papers formula :
# pos: the position in the sequence (row index).
# i: goes from 0 to d_model/2 -1 = 127
# Sine and cosine are 90° phase-shifted.
    # If you used only sine, two different positions could collapse to the same value (e.g., sin(0)=sin(π)=0).
    # With both, you always get a unique 2D representation for each frequency scale:
    # at pos=0 → (sin(0), cos(0)) = (0, 1)
    # at pos=π → (sin(π), cos(π)) = (0, -1)
    # → distinguishable.
    # So: Yes, they use the same input values — that’s exactly the design. One is sin(θ), the other is cos(θ),
    # giving two orthogonal signals per frequency.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000): # allowing a maximum of 5,000 positions
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE) #(5000,256) # hold the positional encodings for every position up to max_len
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1) #(5000,1)
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE)* -(math.log(10000.0) / d_model)) # (128).  2i= (0,2,4,6,...254)
        pe_pos = torch.mul(position, div_term) #Broadcasting (5000,128)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0) #(1,5000,256)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False) #Adds positional encoding (bathches, x.size(1)= seq_len, d_model=256) to word embedding and they are fixed values no gradients flow
        out = self.dropout(x) 
        return out # (batch_size =128, seq_len, token_embedded_dim = 256)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) # the last dimension of the tensor (here, the embedding dimension d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # query @ key.T → (128, 8, 10, 32) @ (128, 8, 32, 10) → (128, 8, 10, 10)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # broadcasting-> (128, 8, 10, 10)
    p_attn = nn.functional.softmax(scores, dim=-1) # (128, 8, 10, 10): softmax is applied along the last dimension (across all keys for each query).
    if dropout is not None:
        p_attn = dropout(p_attn) # dropout randomly zeros out some attention weights with probability, This prevents the model from becoming too dependent on specific attention links, encouraging it to spread focus and generalize better. (happens during training only)
    return torch.matmul(p_attn, value), p_attn 
            #(128, 8, 10, 32) , (128, 8, 10, 10)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1): # (h: number of heads, d_model: embedding_dimention)
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h # 256 // 8 = 32 
        self.h = h # 8
        self.linears = nn.ModuleList([deepcopy(nn.Linear(d_model, d_model)) for i in range(4)]) # 4 linear layers: one each for Q, K, V, and one final output projection after attention. With deepcopy, each is a fresh copy with its own weights and biases.
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None): # Suppose input is (batch=128, seq_len=10, d_model=256)
        if mask is not None:
            mask = mask.unsqueeze(1) # if trg mask -> (128, 1, 10, 10), or if it's src mask -> (128, 1, 1, 10) 
        nbatches = query.size(0)  
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) #  query = (batches=128, heads=8, seq_len=10, d_k=32), key=(1, 8, 10, 32), value=(1, 8, 10, 32)
                            for l, x in zip(self.linears, (query, key, value))] #zip(self.linears, (query, key, value) pairs the first 3 linear layers with query, key, value. So this iterates 3 times only.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # x was (128, 8, 10, 32) → transpose → (128, 10, 8, 32). view(1, 10, 256) → concatenate all heads back together.
        output = self.linears[-1](x)
        return output  # final conextualized input embeddings vectors with positional encoding (batches=128, seq_len=10, d_model= 256)
# nn.Linear(in_features, out_features) expects the last dimension of the input tensor to match in_features.

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        out = self.proj(x)
        probs = nn.functional.log_softmax(out, dim=-1)
        return probs   #(batch_size, seq_len, vocab)
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1): #d_ff: the dimensionality of the feed-forward layer 4*256=1024, d_model: the model’s dimension size = 256.
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.w_1(x)
        h2 = self.dropout(h1)
        return self.w_2(h2)   
    

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') # using KL-divergence loss to compare predicted distribution vs smoothed target distribution.
        self.padding_idx = padding_idx # index for padding tokens (they should not contribute to the loss)
        self.confidence = 1.0 - smoothing # the weight given to the true class (e.g. 0.9 if smoothing=0.1)
        self.smoothing = smoothing #  small value (e.g. 0.1) that controls how much we soften one-hot labels.
        self.size = size # size → number of classes in the output (target vocab size).
        self.true_dist = None # will hold the smoothed label distribution for each batch.

# x → predictions from the model log(softmax), shape (batch_size * seq_length, vocab_size).
# target → true class indices (shape (batch_size*seq_len,)).
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone() # creates a new independent tensor that has the same values as x, but without keeping the computational history of x
        true_dist.fill_(self.smoothing / (self.size - 2)) # Initialize everything with the smoothing value = 0.2. Example: vocab=5, smoothing=0.1 → each wrong class gets 0.1 / (5-2) = 0.0333. Why self.size - 2? Because we exclude padding and the true label itself.
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # Places the true label confidence (e.g. 0.9) at the correct class index.
        true_dist[:, self.padding_idx] = 0 # Force probability of padding token to 0 in every distribution.
        mask = torch.nonzero(target.data == self.padding_idx) # Find all examples in the batch where the target label is padding. mask contains the row indices when the target is padding tokens in the batch 
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0) #If any exist, set their entire row distribution to 0 (so they don’t affect the loss).
        self.true_dist = true_dist
        output = self.criterion(x, true_dist.clone().detach()) # Compute KL-divergence between predicted x (log-probabilities) and smoothed labels. detach() ensures no gradients flow through the targets.
        return output #the scalar batch loss
# Kullback–Leibler (KL) divergence is a way of measuring how one probability distribution Q (the predicted distribution) is
# different from another probability distribution P (the true distribution). DKL​(P∥Q)=i∑​P(i)logQ(i)P(i)​
# when target is 0 the loss = 0

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    # __call__ is basically: “make my object behave like a function".
    # loss_compute = SimpleLossCompute(generator, criterion, opt)
    # loss = loss_compute(x, y, norm)
    def __call__(self, x, y, norm): 
        x = self.generator(x) # x:(128, tgt_seq_len-1, vocab_size) log probablilties 
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), # x:(128 * trg_seq_len-1, vocab_size) . Label smooth of y and compute loss
                            y.contiguous().view(-1)) / norm # y:(128 * trg_seq_len-1,) 
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

# implementation of the Noam learning rate scheduler
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer # the optimizer (e.g., Adam)
        self._step = 0   # counter for training steps (one step corresponds to one optimizer update, which happens after processing one batch of data.)
        self.warmup = warmup # number of warmup steps
        self.factor = factor # scaling factor for learning rate
        self.model_size = model_size # d_model = 256
        self._rate = 0 # current learning rate

    def step(self):
        self._step += 1 #  increase step count
        rate = self.rate() # compute learning rate
        for p in self.optimizer.param_groups: 
            p['lr'] = rate # ==>> update optimizer’s learning rate
        self._rate = rate # # save current rate
        self.optimizer.step()

# For early steps (step < warmup): The step * warmup^(-1.5) term dominates → learning rate increases linearly with steps
# For later steps (step > warmup): The step^(-0.5) term dominates → learning rate decays proportionally to 1/√step.
# This gives a "warmup then decay" shape:
# 📈 Warmup phase: learning rate grows linearly.
# 📉 Decay phase: learning rate decreases as training progresses.
    def rate(self, step=None): # Noam schedule formula
        if step is None:
            step = self._step
        output = self.factor * (self.model_size ** (-0.5) *
        min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return output
