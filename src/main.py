import torch
import torch.optim as optim

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


discriminator = DiscriminatorModel().to(config.device)
generator = GeneratorModel().to(config.device)

# network_test(discriminator, generator)

d_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)

for e in range(config.epochs):

    names = prepare_names()
    num_batches = len(names) // config.batch_size
    print(names[:10])
    d_error = 0
    g_error = 0

    # Decrease the learning rate after 3 iterations
    if e == -3:
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

    print(f"Finished with epoch {e}")
    print(f"Discriminator error: {d_error}")
    print(f"Generator error: {g_error}")

    generated = utils.tensor_to_word(generator(samples=10))
    print(f"Samples: {generated}")
