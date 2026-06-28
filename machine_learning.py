import torchvision

# This one line downloads everything automatically!
train_data = torchvision.datasets.CIFAR100(
    root='./data',    # saves to a folder called 'data'
    train=True,       # download training set (50,000 images)
    download=True     # yes, please download it
)

test_data = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,      # download test set (10,000 images)
    download=True
)
