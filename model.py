import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, emb_size, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()

        self.emb = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        output, (hidden, cell) = self.lstm(x)

        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, emb_size, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x: (batch_size)
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedding = self.embedding(x)  # (batch_size, 1, emb_size)

        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch_size, output_size)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):

        batch_size = source.size(0)
        target_seq_length = target.size(1)
        target_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, target_seq_length, target_vocab_size).to(source.device)
        encoder_output, hidden, cell = self.encoder(source)
        # encoder_output shape: (batch_size, source_seq_length, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        inputs = target[:, 0]  # (batch_size)
        for t in range(1, target_seq_length):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[:, t, :] = output
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                inputs = target[:, t]
            else:
                top1 = output.argmax(1)
                inputs = top1
        return outputs
