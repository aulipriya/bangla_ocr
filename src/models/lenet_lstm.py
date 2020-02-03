import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, number_of_input, number_of_hidden, nunmer_of_out):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(number_of_input, number_of_hidden, bidirectional=True)
        self.embedding = nn.Linear(number_of_hidden * 2, nunmer_of_out)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class LenetLstm(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(LenetLstm, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('conv{0}'.format('0'),
                       nn.Conv2d(3, 6, kernel_size=(5, 5)))
        self.cnn.add_module('tanh{0}'.format('0'), nn.Tanh())
        self.cnn.add_module('pooling{0}'.format('0'), nn.AvgPool2d(kernel_size=(2, 2), stride=2))

        self.cnn.add_module('conv{0}'.format('1'),
                            nn.Conv2d(6, 16, kernel_size=(5, 5)))
        self.cnn.add_module('tanh{0}'.format('1'), nn.Tanh())
        self.cnn.add_module('pooling{0}'.format('1'), nn.AvgPool2d(kernel_size=(2, 2), stride=2))
        self.cnn.add_module('conv{0}'.format('2'),
                            nn.Conv2d(16, 120, kernel_size=(5, 5)))
        self.cnn.add_module('tanh{0}'.format('2'), nn.Tanh())

        self.fc = nn.Sequential()
        self.fc.add_module('fc0', nn.Linear(120, 32))
        self.fc.add_module('fcTanh0', nn.Tanh())
        self.fc.add_module('fc1', nn.Linear(32, 64))
        self.fc.add_module('fcTanh1', nn.Tanh())
        self.rnn = nn.Sequential(
            BidirectionalLSTM(2, 128, 128),
            BidirectionalLSTM(128, 32, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print('size after conv')
        print(conv.size())

        conv = conv.view(input.size(0), -1)
        print(conv.size())

        fc = self.fc(conv)
        print(fc.size())

        fc = fc.view(input.size(0), 32, 2)
        fc = fc.permute(1, 0, 2)
        print(fc.size())

        output = self.rnn(fc)
        return output
        # conv = conv.squeeze(2)
        # # print('size after squeeze')
        # # print(conv.size())
        # conv = conv.permute(2, 0, 1)  # [w, b, c]
        # # rnn features
        # output = self.rnn(conv)
        # print("from model {}".format(output))
        # return output

