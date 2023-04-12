import pandas as pd
import re
import torch
import torch.optim as optim
from vocabulary import Vocabulary
from dataset import English2French
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from model import *
import matplotlib.pyplot as plt
from tqdm import tqdm


def remove_punct(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)


def text_preprocessing(text):
    text = text.lower()
    text = remove_punct(text)
    return text


def collate_fn(batch):
    eng = [x[0] for x in batch]
    fr = [x[1] for x in batch]
    eng = pad_sequence(eng, batch_first=True)
    fr = pad_sequence(fr, batch_first=True)
    return eng, fr


def train(model, optimizer, criterion, train_loader):
    model.train()
    losses = []
    epoch = 10
    for _ in range(epoch):
        epoch_loss = 0
        for input_batch, target_batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            output_batch = model(input_batch, target_batch)
            output_batch = output_batch[1:].view(-1, output_batch.shape[-1])
            target_batch = target_batch[1:].view(-1)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses += epoch_loss
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Loss')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('eng-french.csv')
    data['eng'] = data['English words/sentences'].apply(text_preprocessing)
    data['fr'] = data['French words/sentences'].apply(text_preprocessing)
    data.drop(['English words/sentences', 'French words/sentences'], axis=1, inplace=True)

    dataset = English2French(data)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2, collate_fn=collate_fn)

    learning_rate = 0.001
    input_size_encoder = len(dataset.english)
    input_size_decoder = len(dataset.french)
    output_size = len(dataset.french)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(encoder_embedding_size, input_size_encoder, hidden_size,
                      num_layers, enc_dropout).to(device)

    decoder = Decoder(decoder_embedding_size, input_size_decoder, hidden_size,
                      output_size, num_layers, dec_dropout).to(device)

    pad_idx = dataset.english.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, optimizer, criterion, dataloader)
