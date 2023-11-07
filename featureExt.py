class EncoderModel(nn.Module):

  def __init__(self, vocab_size, dim, pad_ind):

    # Word embeddings
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, padding_idx=pad_ind)

    # Bi-LSTM
    self.bi_lstm = nn.LSTM(input_size=dim, bias=True, bidirectional=True)

  def forward(self, sentence):

    embedded = self.embedding(sentence)
    output = self.bi_lstm(embedded)

    return output
