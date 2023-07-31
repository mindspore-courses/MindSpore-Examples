import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.dataset import Mnist
from mindspore.train import Model,LearningRateScheduler
from mindvision.engine.callback import LossMonitor
import argparse

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,pad_mode="pad")
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,pad_mode="pad")
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Dense(12544,128)
        self.fc2 = nn.Dense(128,10)

    def construct(self,x):
        x = self.conv1(x)
        x = ops.relu(x)
        x = self.conv2(x)
        x = ops.relu(x)
        x = ops.max_pool2d(x,2)
        x = self.dropout1(x)
        x = ops.flatten(x,start_dim = 1)
        x = self.fc1(x)
        x = ops.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = ops.log_softmax(x,axis=1)
        return output

def get_parser():
    parser = argparse.ArgumentParser(description='Mindspore MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    #默认值是14!!!记得改回去!!!!!!!!!!!!!!!!!
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args
    
def learning_rate_function(lr, cur_step_num):
    if cur_step_num%1875 == 0:
        lr = lr * args.gamma
    return lr 

def train(args, model, train_loader):
    model.train(epoch=args.epochs, train_dataset=train_loader, callbacks=[LearningRateScheduler(learning_rate_function),LossMonitor(0.01,1875)])
     

def test(model,test_loader):
    acc = model.eval(test_loader)
    print("Accuracy: ",acc)
    

def main(args):
    if args.no_gpu:
        ms.set_context(device_target="CPU")
    ms.set_seed(args.seed)

    # Load data
    download_train = Mnist(path="../data/mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True)
    download_eval = Mnist(path="../data/mnist", split="test", batch_size=32, resize=32, download=True)
    dataset1 = download_train.run() #Train
    dataset2 = download_eval.run() #Test
    model = Net()

    optimizer =  nn.optim.Adadelta(model.trainable_params(), learning_rate=args.lr, rho=0.9, weight_decay=0.0)
    criterion = nn.NLLLoss()
    train_model = Model(network=model, loss_fn=criterion, optimizer=optimizer,metrics={'accuracy'})

    train(args, train_model,dataset1)
    test(train_model, dataset2)

    if args.save_model:
        ms.save_checkpoint(model, "Minist.ckpt")


if __name__ == '__main__':
    args = get_parser()
    main(args)