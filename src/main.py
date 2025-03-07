import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from network import DiscriminatorModel, GeneratorModel
from config import config
from data import prepare_names
from training import train_discriminator, train_generator
import utils


'''
A quick sanity check to see if the network logic is working
'''


def network_test(discriminator, generator):
    test_word = utils.pad_word('alma')
    test_input = utils.word_to_tensor(test_word) \
        .to(config.device).type(utils.float_type)[None, :].transpose(0, 1)

    print("Testing the discriminator")
    print(test_input)
    print(discriminator(test_input))

    print("Testing the generator")
    samples = generator(samples=2)
    samples = samples.transpose(0, 1)
    print(samples)
    print(samples.size())
    # print(utils.tensor_to_word(samples))
    # print(generator.log_likelihood(test_input))
    print(generator.log_likelihood(samples))


def run():
    discriminator = DiscriminatorModel().to(config.device)
    generator = GeneratorModel().to(config.device)

    # network_test(discriminator, generator)

    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=config.learning_rate)
    g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)

    d_errors = np.zeros(config.epochs)
    g_errors = np.zeros(config.epochs)

    for e in range(config.epochs):

        names = prepare_names()
        num_batches = len(names) // config.batch_size
        # print(names[:10])
        d_error = 0
        g_error = 0

        # Decrease the learning rate after 3 iterations
        if e == 5:
            for param in d_optimizer.param_groups:
                param['lr'] = config.learning_rate / 10
            for param in g_optimizer.param_groups:
                param['lr'] = config.learning_rate / 10

        for b in range(num_batches):

            # print(f"Starting batch {b}")

            # 1 Generate Real data
            batch_names = names[b*config.batch_size: (b+1)*config.batch_size]
            real_data = torch.stack([
                utils.word_to_tensor(word, pad=True) for word in batch_names
            ]).to(config.device).transpose(0, 1).type(utils.float_type)

            # 2 Generate Fake data
            fake_data = generator(samples=config.batch_size).transpose(0, 1)

            # 3 Train discriminator
            d_batch_error, prediction_real, _ =\
                train_discriminator(d_optimizer, discriminator,
                                    real_data, fake_data)

            # 4 Train generator
            g_batch_error = train_generator(
                g_optimizer, discriminator, generator, real_data)

            d_error += d_batch_error
            g_error += g_batch_error
            # print(f"d_batch_error={d_batch_error}")
            # print(f"g_batch_error={g_batch_error}")

        d_error /= num_batches
        g_error /= num_batches
        d_errors[e] = d_error
        g_errors[e] = g_error

        print(f"Finished with epoch {e}")
        # print(f"Discriminator error: {d_error}")
        # print(f"Generator error: {g_error}")

        generated = utils.tensor_to_word(generator(samples=10))
        # print(f"Samples: {generated}")

    return d_errors, g_errors, discriminator, generator


# discriminator = DiscriminatorModel().to(config.device)
# generator = GeneratorModel().to(config.device)
# network_test(discriminator, generator)

# all_d_errors = []
# all_g_errors = []
# for it in range(config.runs):
#     d_errors, g_errors, _, _ = run()
#     print("=====================")
#     print(f"Finished with run {it}")
#     print("=====================")
#     all_d_errors.append(d_errors)
#     all_g_errors.append(g_errors)

# d_df = pd.DataFrame(np.array(all_d_errors))
# g_df = pd.DataFrame(np.array(all_g_errors))
# d_df.to_csv(config.log_folder + 'discriminator.csv')
# g_df.to_csv(config.log_folder + 'generator.csv')


def _write_samples(samples, file):
    f = open(file, 'w')
    for sample in samples:
        f.write(sample + '\n')
    f.close()


_, _, _, generator = run()
short_samples = utils.tensor_to_word(
    generator(samples=200, end_char_modifier=1.0))
long_samples = utils.tensor_to_word(
    generator(samples=200, end_char_modifier=0.0))

_write_samples(short_samples, config.log_folder + 'short_samples.txt')
_write_samples(long_samples, config.log_folder + 'long_samples.txt')
