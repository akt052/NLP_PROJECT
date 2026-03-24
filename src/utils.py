from collections import Counter

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_vocab(data, field, min_freq=2):
    counter = Counter()
    for item in data:
        counter.update(item[field])

    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def numericalize(tokens, vocab):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]