import torch.nn as nn
from models.mobilentv2 import MobileNetV2


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


class MobilenetRNN(nn.Module):

    def __init__(self, nc, nclass, nh, n_rnn=2):
        super(MobilenetRNN, self).__init__()
        cnn = MobileNetV2(input_channels=nc, num_classes=nclass)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # print('size before squeeze')
        # print(conv.size())
        conv = conv.squeeze(2)
        # print('size after squeeze')
        # print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # print("from model {}".format(conv.size()))
        # rnn features
        output = self.rnn(conv)

        return output
