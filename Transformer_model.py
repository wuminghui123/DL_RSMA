#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import copy
#=======================================================================================================================
#=======================================================================================================================
class TRANS_BLOCK(nn.Module): #输入信道输出 量化后的B比特反馈信息
    def __init__(self,src_vocab_size,out_vocab_size,model_dimension,number_of_layers):
        super(TRANS_BLOCK,self).__init__()
        # model_dimension = 256
        dropout_probability = 0.1
        number_of_heads = 8
        log_attention_weights = False
        # number_of_layers = 6
        self.src_embedding = Embedding(src_vocab_size, model_dimension)  # 对输入进行embedding
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        self.trans_encoder = Trans_Encoder(encoder_layer, number_of_layers, out_vocab_size)
        
        
    def forward(self,x_ini):
        
        src_embeddings_batch = self.src_embedding(x_ini)  # get embedding vectors for src token ids
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)  # add positional embedding
        out = self.trans_encoder(src_embeddings_batch, src_mask=None)  # forward pass through the encoder
        return out


class Encoder(nn.Module):
    num_quan_bits = 2
    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        src_vocab_size = 64
        model_dimension = 384
        dropout_probability = 0.1
        number_of_heads = 8
        log_attention_weights = False
        number_of_layers = 6
        self.src_embedding = Embedding(src_vocab_size, model_dimension)  # 对输入进行embedding
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        self.trans_encoder = Trans_Encoder(encoder_layer, number_of_layers, src_vocab_size)

        self.fc = nn.Linear(768, int(feedback_bits / self.num_quan_bits))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.num_quan_bits)
    def forward(self, x):
        x = x.contiguous().view(-1,12,64)
        #x = x.permute(0,3,1,2)

        src_embeddings_batch = self.src_embedding(x)  # get embedding vectors for src token ids
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)  # add positional embedding
        out = self.trans_encoder(src_embeddings_batch, src_mask=None)  # forward pass through the encoder
        out = out.contiguous().view(-1,int(12*32*2))
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)
        return out

class Decoder(nn.Module):
    num_quan_bits = 2
    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        src_vocab_size = 64
        model_dimension = 384
        dropout_probability = 0.1
        number_of_heads = 8
        log_attention_weights = False
        number_of_layers = 6

        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.num_quan_bits)
        self.fc = nn.Linear(int(feedback_bits / self.num_quan_bits), 768)

        # self.trg_embedding = Embedding(trg_vocab_size, model_dimension)
        # self.trg_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        # mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        # pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        # decoder_layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)
        # self.trans_decoder = Trans_Decoder(decoder_layer, number_of_layers, trg_vocab_size)
        # self.decoder_generator = DecoderGenerator(model_dimension, trg_vocab_size)

        self.src_embedding = Embedding(src_vocab_size, model_dimension)  # 对输入进行embedding
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        self.trans_encoder = Trans_Encoder(encoder_layer, number_of_layers, src_vocab_size)

    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits / self.num_quan_bits)) - 0.5
        out = self.fc(out)
        out = out.view(-1, 12, 64)

        out = self.src_embedding(out)  # get embedding vectors for trg token ids
        out = self.src_pos_embedding(out)  # add positional embedding
        # Shape (B, T, D), where B - batch size, T - longest target token-sequence length and D - model dimension
        out = self.trans_encoder(out, src_mask=None)

        # After this line we'll have a shape (B, T, V), where V - target vocab size, decoder generator does a simple
        # linear projection followed by log softmax
        #trg_log_probs = self.decoder_generator(trg_representations_batch)

        # Reshape into (B*T, V) as that's a suitable format for passing it into KL div loss
        #trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1])

        #out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(-1, 768)
        return out



def Mish(x):
    x = x * torch.tanh(F.softplus(x))
    return x


# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)
def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Encoder and Decoder Class Defining
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)



class Trans_Encoder(nn.Module):

    def __init__(self, encoder_layer, number_of_layers, src_vocab_size):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)
        self.linear = nn.Linear(encoder_layer.model_dimension, src_vocab_size)

    def forward(self, src_embeddings_batch, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become (the initial encoder layer
        # has embedding vectors as input but later layers have richer token representations)
        src_representations_batch = src_embeddings_batch

        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        src_representations_batch = self.norm(src_representations_batch)
        out = self.linear(src_representations_batch)

        return out


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention # 由多个self attention 并行组成
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        # 单行函数定义，srb为输入,
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch




#
# Decoder architecture
#


class Trans_Decoder(nn.Module):

    def __init__(self, decoder_layer, number_of_layers,trg_vocab_size):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)
        self.linear = nn.Linear(decoder_layer.model_dimension, trg_vocab_size)

    def forward(self, trg_embeddings_batch, trg_mask):
        # Just update the naming so as to reflect the semantics of what this var will become
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            # Target mask masks pad tokens as well as future tokens (current target token can't look forward)
            trg_representations_batch = decoder_layer(trg_representations_batch, trg_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        trg_representations_batch = self.norm(trg_representations_batch)
        out = self.linear(trg_representations_batch)
        return out


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, trg_representations_batch, trg_mask):
        # Define anonymous (lambda) function which only takes trg_representations_batch (trb - funny name I know)
        # as input - this way we have a uniform interface for the sublayer logic.
        # The inputs which are not passed into lambdas are "cached" here that's why the thing works.
        decoder_trg_self_attention = lambda trb: self.multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)

        # Self-attention MHA sublayer followed by a source-attending MHA and point-wise feed forward net sublayer
        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, self.pointwise_net)

        return trg_representations_batch

class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

        self.init_params()

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#


# Note: the original paper had LayerNorm AFTER the residual connection and addition operation
# multiple experiments I found showed that it's more effective to do it BEFORE, how did they figure out which one is
# better? Experiments! There is a similar thing in DCGAN and elsewhere.
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))


class DecoderGenerator(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super().__init__()

        self.linear = nn.Linear(model_dimension, vocab_size)

        # -1 stands for apply the log-softmax along the last dimension i.e. over the vocab dimension as the output from
        # the linear layer has shape (B, T, V), B - batch size, T - max target token-sequence, V - target vocab size
        # again using log softmax as PyTorch's nn.KLDivLoss expects log probabilities (just a technical detail)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        # Project from D (model dimension) into V (target vocab size) and apply the log softmax along V dimension
        return self.log_softmax(self.linear(trg_representations_batch))


class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.

        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this auto-magically behind the scenes.

    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))


class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic/robust as possible whereas
        on the other hand I only want to support a limited use-case.

        Also this is arguable the most important architectural component in the Transformer model.

        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:

        https://jalammar.github.io/illustrated-transformer/

        Optimization notes:

        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 "feed forward nets" (without activation/identity hence the quotation marks).
        Conceptually both implementations are the same.

        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).

        This has an important optimization implication, they can reshape their matrix into (B*NH, S/T, HD)
        (where B - batch size, S/T - max src/trg sequence length, NH - number of heads, HD - head dimension)
        in a single step and I can only get to (B, NH, S/T, HD) in single step
        (I could call contiguous() followed by view but that's expensive as it would incur additional matrix copy)

    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        # 3*8*2*2
        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
            # 即将mask==0的替换为-1e9,其余不变

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        # 3*8*2*64
        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)
        # 3*2*512
        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


class Embedding(nn.Module): # embedding模块

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        #self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.linear = nn.Linear(vocab_size, model_dimension)
        self.model_dimension = model_dimension # 表示embedding的维度，即词向量的维度
        # vocab size 表示词汇表的数量

    def forward(self, token_ids_batch):  # 输入
        #assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        #embeddings = self.embeddings_table(token_ids_batch)  # 输入embedding后的矢量
        embeddings = self.linear(token_ids_batch)
        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)  # 这个操作不知道为什么


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings) # 整个Embedding是Word Embedding与Positional Encoding直接相加之后的结果


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])





def NMSE(x_hat,x):
    x = torch.reshape(x, (len(x), -1))
    x_hat = torch.reshape(x_hat, (len(x_hat), -1))
    
    power = torch.sum(abs(x) ** 2, axis=1)
    mse = torch.sum(abs(x - x_hat) ** 2, axis=1)
    nmse = torch.mean(mse / power)
    return nmse

