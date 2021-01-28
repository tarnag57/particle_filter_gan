import torch
import torch.nn.functional as F
import copy

from config import config

letters = ['.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '$']


float_type = torch.FloatTensor if config.device == 'cpu' else torch.cuda.FloatTensor
long_type = torch.LongTensor if config.device == 'cpu' else torch.cuda.LongTensor


##################################################
# HELPER FUNCTIONS FOR ONE-HOT ENCODING OF LETTERS
##################################################


def char_to_int(c):
    c = str.lower(c)
    if c == '.':
        return 0
    if c == '$':
        return 27
    # [a-z]
    if 96 < ord(c) and ord(c) < 123:
        return ord(c) - 96
    else:
        raise Exception(f"Character is not in vocab; c={c}")


def int_to_char(i):
    if i < len(letters):
        return letters[i]
    else:
        raise Exception(f"Int is not in vocab range; i={i}")


# Quick sanity check
for i, c in enumerate(letters):
    assert(i == char_to_int(c))
    assert(c == int_to_char(i))


def pad_word(word, dotted=False):
    new_word = copy.deepcopy(word)
    diff = 15 - len(new_word)
    for _ in range(diff):
        new_word += '$'

    if dotted:
        new_word = '.' + new_word

    return new_word


def word_to_tensor(word, pad=False):
    if pad:
        word = pad_word(word)
    return F.one_hot(
        torch.Tensor([char_to_int(c) for c in word]).type(long_type),
        num_classes=len(letters)
    )


def tensor_to_word(tensor):
    names = []
    for t in tensor:
        name = []
        for c in t:
            other = torch.arange(len(letters))
            char = int(torch.dot(c.type(torch.LongTensor), other))
            name.append(letters[char])

        names.append("".join(name))
    return names
