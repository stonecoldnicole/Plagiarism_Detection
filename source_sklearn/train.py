from __future__ import print_function

import argparse
import os
import pandas as pd

## TODO: Import any additional libraries you need to define a model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from model import DetectorNet

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # Code below is reused from the Moon Data Class Exercise
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], 
                      model_info['hidden_dim'], 
                      model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)


# Load the training data from a csv file
# Code below is reused from the Moon Data Class Exercise
def _get_train_loader(batch_size, data_dir):
    print("Getting data loader.")

    # read in csv file
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, names=None)

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # features are the rest
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)


def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum hyperparameter for SGD model (default: 0.9)')
    parser.add_argument('--input_dim', type=int, default=2, metavar='IN',
                        help='number of input features for model (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=7, metavar='H',
                        help='number of hidden dimensions for model (default: 7)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='number of dimensions output by model (default: 1)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    # set device to gpu if available, otherwise cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.maual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Read in csv training file
    training_dir = args.data_dir
    training_loader = _get_train_loader(args.batch_size, args.data_dir)
#     train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

#     # labels are first column
#     train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
#     # features are the rest
#     train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()
    
    
    ## --- Your code here --- ##
    # Define a model & set hyperparameters
    model = DetectorNet(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    save_model_params(model, args.model_dir)
    
    # Define training loss function and optimizer
    criterion = nn.BCELoss() # model returns only one value, cross entropy loss won't work
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
    ## TODO: Train the model **This line also saves the trained model**
    train(model, train_loader, args.epochs, optimizer, criterion, device)
    
    ## --- End of your code  --- ##