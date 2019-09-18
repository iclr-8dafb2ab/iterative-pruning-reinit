{
    ('conv1', 'weight'): partial(torch.nn.init.kaiming_uniform_, nonlinearity='relu'),
    ('conv2', 'weight'): partial(torch.nn.init.kaiming_uniform_, nonlinearity='relu'),
    ('fc1', 'weight'): partial(torch.nn.init.kaiming_uniform_, nonlinearity='relu'),
    ('fc2', 'weight'): partial(torch.nn.init.kaiming_uniform_, nonlinearity='relu'),
    ('fc3', 'weight'): partial(torch.nn.init.kaiming_uniform_, nonlinearity='relu')
}