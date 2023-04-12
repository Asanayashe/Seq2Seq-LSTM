from torch.utils.data import Dataset
from vocabulary import Vocabulary
import torch


class English2French(Dataset):

    def __init__(self, df, freq_threshold=5):
        self.df = df

        self.english = Vocabulary(freq_threshold)
        self.english.build_vocabulary(self.df.eng.to_list())

        self.french = Vocabulary(freq_threshold)
        self.french.build_vocabulary(self.df.fr.to_list())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        eng_txt = []
        eng_txt += [self.english.stoi["<SOS>"]]
        eng_txt += self.english.numericalize(self.df.eng[idx])
        eng_txt += [self.english.stoi["<EOS>"]]

        fr_txt = []
        fr_txt += [self.french.stoi["<SOS>"]]
        fr_txt += self.french.numericalize(self.df.fr[idx])
        fr_txt += [self.french.stoi["<EOS>"]]

        return torch.tensor(eng_txt), torch.tensor(fr_txt)
