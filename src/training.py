import torch


def train_discriminator(optimizer, discriminator, real_data, fake_data):
    N = len(real_data)
    # Reset gradients
    optimizer.zero_grad()

    # 1 Prediction on Real Data
    prediction_real = discriminator(real_data)

    # 2 Prediction on Fake Data
    prediction_fake = discriminator(fake_data)

    # 3 Calculate error and backprop
    error = torch.log(prediction_real) / N + torch.log(1-prediction_fake) / N
    error = -torch.sum(error)        # Negate error since we are maximising
    error.backward()

    # 4 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error, prediction_real, prediction_fake


def train_generator(optimizer, discriminator, generator, real_data):
    N = len(real_data)
    # Reset gradients
    optimizer.zero_grad()

    # 1 Calculate the likelihoods for the generator producing the real data
    log_likelihoods = generator.log_likelihood(real_data)
    likelihoods = torch.exp(log_likelihoods)

    # 2 Calulate loss
    prediction_real = discriminator(real_data)
    loss_vector = torch.log(1 - prediction_real) * likelihoods
    error = torch.mean(loss_vector)
    error = - torch.log(-error)
    error.backward()

    # 3 Train on loss
    optimizer.step()

    return error
