import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, tokenize):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.tokenize = tokenize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        src_tokens = self.tokenize(item["src"])
        trg_tokens = self.tokenize(item["trg"])

        src = [self.src_vocab.get(t, self.src_vocab["<unk>"]) for t in src_tokens]
        trg = [self.trg_vocab["<sos>"]] + \
              [self.trg_vocab.get(t, self.trg_vocab["<unk>"]) for t in trg_tokens] + \
              [self.trg_vocab["<eos>"]]

        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch, src_pad_idx, trg_pad_idx):
    src_batch, trg_batch = zip(*batch)

    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    trg_pad = pad_sequence(trg_batch, batch_first=True, padding_value=trg_pad_idx)

    return src_pad, trg_pad