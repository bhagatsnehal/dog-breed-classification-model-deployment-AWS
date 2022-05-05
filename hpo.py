# Import your dependencies.

import numpy as np
import io
import sys
import logging
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



# Initialize Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

    
def test(model, test_loader, criterion, device):
    '''
    This function can take a model and a 
          testing data loader and will get the test accuray of the model
    '''

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        logger.debug(f'Length of the TEST data loader  is: {len(test_loader)}')
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            logger.debug(f'Target Shape is: {target.shape}')
            logger.debug(f'Output Shape is: {output.shape}')
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Loss: {:.0f}%\n".format(test_loss)
        )
    logger.info(
        "Test set: Accuracy: {:.0f}% ({}/{})\n".format(
            100.0 * correct / len(test_loader.dataset), correct, len(test_loader.dataset)
        )
    )
    
    pass


def train(model, train_loader, criterion, optimizer, epoch, device):
    '''
    This function can take a model and
          data loaders for training and will train the model
    '''

    model.train()
    for e in range(epoch):
        running_loss=0
        correct=0
        for data, target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
             Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    pass

    
def net():
    # Complete this function that initializes model to use a pretrained model
    
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    # Initialize a model by calling the net function

    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model.to(device)
    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = ImageFolder(args.data_dir, transform=training_transform)
    testset = ImageFolder(args.test_data_dir, transform=testing_transform)

    logger.info("Batch Size {}".format(args.batch_size))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    
    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    # Call the train function to start training your model, training data fetched from S3 using channel
    train(model, train_loader, loss_criterion, optimizer,args.epochs, device)
    
    # Test the model to see its accuracy
    test(model, test_loader, loss_criterion, device)
    
    # Save the trained model
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify all the hyperparameters needed to train the model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    args=parser.parse_args()
    
    main(args)
