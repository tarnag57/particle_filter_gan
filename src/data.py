import torch

from config import config


def get_names_from_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()

    # We are only including names that contain letters [a-z]
    names = []
    for line in lines:
        if line[:-1].isalpha() and len(line) <= config.max_len:
            name = line[:-1] + "$"    # Discard '\n' and append '$'
            names.append(name)

    return names

# Use torch for permutation to have a unique random seed


def shuffle(dataset):
    indicies = torch.randperm(len(dataset))
    names = []
    for i in indicies:
        names.append(dataset[i])
    return names


def prepare_names():
    prefix = config.data_folder
    females = get_names_from_file(prefix + 'female.txt')
    males = get_names_from_file(prefix + 'male.txt')
    return shuffle(females + males)
