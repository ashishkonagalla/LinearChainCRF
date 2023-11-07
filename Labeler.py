class Labeler(nn.Module):

  def __init__(self, vocab_input, vocab_target, emb_dim, hid_dim):

    super(Tagger, self).__init__()

    # Extracting features from the input sequence
    self.encoder = EncoderModel(vocabuary_size=vocab_input.size, embedding_dim=emb_dim,
                               hidden_dimension=hid_dim, padding_idx=vocab_input.pad_idx)

    # Linear projection (parameters W and b)
    self.formTags = nn.Linear(hid_dim *2, vocab_target.size)

    # Call the linear-chain CRF here
    # self.labeler = myCRF(parameters)

  def forward(self, input_seq, target_seq):

    # input_sequence 
    lstm_features = self.encoder(input_seq)
    crf_features = self.formTags(lstm_features)

    # compute loss, score using the call to myCRF and return loss, score. 
