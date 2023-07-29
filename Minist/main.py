import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.dataset import Mnist
import argparse

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1,pad_mode="pad")
        self.conv2 = nn.Conv2d(32,64,3,1,pad_mode="pad")
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Dense(9216,128)
        self.fc2 = nn.Dense(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = ops.relu(x)
        x = self.conv2(x)
        x = ops.relu(x)
        x = ops.max_pool2d(x,2)
        x = self.dropout1(x)
        x = ops.flatten(x,1)
        x = self.fc1(x)
        x = ops.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = ops.log_softmax(x,axis=1)
        return output
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
#       data, target = data.to(device), target.to(device)
        optimizer.clear_gradients()
        output = model(data)
        loss = ops.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    model.set_train(False)
    for data, target in test_loader:
#       data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += ops.nll_loss(output,target,reduction="sum").asnumpy()
        pred = output.argmax(axis=1)
        correct += pred.eq(target).sum()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def main():
    parser = argparse.ArgumentParser(description='Mindspore MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-GPU', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    if parser.no_GPU:
        ms.set_context(device_target="CPU")

    ms.set_seed(args.seed)
    train_kwargs = {'batch_size':args.batch_size}
    test_kwargs = {'batch_size':args.test_batch_size}

    # Load data
    download_train = Mnist(path="../data/mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True)
    download_eval = Mnist(path="../data/mnist", split="test", batch_size=32, resize=32, download=True)
    dataset_train = download_train.run()
    dataset_eval = download_eval.run()