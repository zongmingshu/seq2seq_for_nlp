
seg = ' '
unregisted = 3
sos = 2
eos = 1
pad = 0

single = 3
begin = 4
inner = 5
end = 6

max_seq_len = 256

auto_encoder = True

bidirection = True
attention = True
hidden_size = 64

# h_(j): encoder output at j slide
# s_(i-1): decoder state at i - 1 slide
# e_(i,j) = W^T*(U^T*s_(i-1) + V^T*h_(j))
# a_(i,j) = softmax(e_(i,j))
# attention_num_units: dimension of U^T, V^T and W^T
attention_num_units = 32

# concatenate the context vector and
# output of the decoder's internal rnn cell
attention_layer_size = 16

keep_prob = 0.9
layer_size = 2

# vocab size + 4
encoder_vocab_size = 4687
if not auto_encoder:
    decoder_vocab_size = 7
else:
    decoder_vocab_size = encoder_vocab_size
label_class_size = 53010

embedding_initializer = 'default_initializer'
embedding_dim = 32
grad_clip = 1.0
time_major = True

label_class_loss_factor = 0.0

num_epoch = 500
epoch_size = 100
batch_size = 128

num_gpu = 2


