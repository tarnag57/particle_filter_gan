import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from config import config
import utils

#########################
# GAN MODELS (LSTM BASED)
#########################


'''
An LSTM-based discriminator model.

Input: The word to be inspected. Real words always end with '$'.
Output: Prediction of the network. 1 = Real, 0 = Fake.
'''


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.input_size = len(utils.letters)
        self.hidden_size = 8
        self.num_layers = 1
        self.final_classes = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.fc = nn.Linear(self.hidden_size, self.final_classes)

    '''
    Feed-forward of the netowrk. We assume that the input is the correct size.

    :param x: The input word with size=(len(word), batch_size, len(letters))
    :return: Prediction with size=(1, batch_size, 1)
    '''

    def forward(self, x):

        x.to(config.device)
        x.requires_grad = False

        output, _state = self.lstm(x)
        final_output = output[-1]
        return torch.sigmoid(self.fc(final_output))


'''
An LSTM-based generator model.

Input: None
Output: Generated name (hopefully ending with '$').
'''


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.input_size = len(utils.letters)
        self.output_size = len(utils.letters)
        self.hidden_size = 8
        self.num_layers = 1         # We need to go through one-by-one to input the selected char

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=2)

    '''
    Generates a name using a fully-connected layer and Softmax on the
    hidden states at each step. Applies a new LSTM layer till '$' is
    generated or config.max_len is reached.

    :param samples int: The number of samples to be generated
    :return char[]: The generated name as an array of characters.
    '''

    def forward(self, samples):

        starting_char = utils.word_to_tensor(
            '.').to(config.device).type(utils.float_type).repeat([samples, 1])[None, :]
        output, state = self.lstm(starting_char)
        gen_name_tensors = []
        next_input = None

        for _ in range(config.max_len):

            # state == (hidden, channel)
            probs = self.softmax(self.fc(output))

            # Sample from the distribution
            distr = Categorical(probs)
            try:
                categories = distr.sample()
            except:
                print("Sampling crashed")
                print(f"Probs: {probs}")
                print(f"Output: {output}")
                print(f"Next_input: {next_input}")
                raise Exception("BOOM!")

            next_input = F.one_hot(
                categories.type(utils.long_type),
                num_classes=len(utils.letters)
            ).to(config.device).type(utils.float_type)
            gen_name_tensors.append(next_input)

            output, state = self.lstm(next_input, state)

        # Annoyingly, torch.stack always introduces a new dimension...
        return torch.stack(gen_name_tensors).squeeze(dim=1).transpose(0, 1)

    '''
    Calculates the log-likelihood of the model generating the word x.

    :param x str: The inspected word (should end with '$').
    :return:
    '''

    def log_likelihood(self, x):

        batch_size = x.size()[1]      # The second dimension is the batch size
        starting_char = utils.word_to_tensor('.').to(config.device) \
            .type(utils.float_type).repeat([batch_size, 1])[None, :]
        output, state = self.lstm(starting_char)

        total_log_lik = torch.zeros(batch_size).to(config.device)

        for layer in x:
            probs = self.softmax(self.fc(output)).transpose(0, 1)
            likelihood = torch.bmm(probs, layer[:, :, None]).flatten()
            total_log_lik += torch.log(likelihood)

            output, state = self.lstm(layer[None, :], state)

        return total_log_lik
