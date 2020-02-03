import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class StrLabelConverter(object):
    def __init__(self):
        self.dictionary = {}
        # for i, char in enumerate(alphabet):
        #     # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        #     self.dict[char] = i + 1


    def get_total_data(self):
        banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎ০১২৩৪৫৬৭৮৯"
        charlist = []  # I have the chars in a list in specific index
        for i in range(0, len(banglachars)):
            charlist.append(banglachars[i])

        # Add modifiers in the character list
        modifiers = "ঁােিীুোৌূৗংঃৃ ্"
        for i in range(0, len(modifiers)):
            charlist.append(modifiers[i])
        # Total = charlist + modlist
        total = charlist
        return total

    def encode_total(self):
        # Encode integer numbers for total dataset
        total = self.get_total_data()
        for i in enumerate(total):
           self.dictionary.update({i[0]+1: i[1]})
        # print(self.dictionary)
        return self.dictionary


    def convert_integer_to_string(self, t, length):
        dictionary = self.encode_total()
        if length.numel() == 1:
            length = length[0]
            # assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
            #                                                                                              length)

            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(dictionary[int(t[i])])
            return ''.join(char_list)
        else:
            # batch mode
            # assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
            #     t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = int(length[i])
                temp_length = index+l
                texts.append(
                    self.convert_integer_to_string(
                        t[index:temp_length].data, torch.IntTensor([l])))
                index += l
            return texts
    def get_key_from_dict_value(self,dict,char):
        for key, value in dict.items():
            if value == char:
                return key

    def convert_string_to_integer(self, text, integer_sequence=[]):
        dictionary = self.encode_total()
        i = 0
        lengths = []
        if isinstance(text, str):
            while i < (len(text)):
                temp = self.get_key_from_dict_value(dictionary, text[i])
                if temp is None:
                    integer_sequence.append(0)
                else:
                    integer_sequence.append(temp)
                i = i + 1
            # print(integer_sequence)
        elif isinstance(text, collections.Iterable):
            for s in text:
                lengths.append(len(s))

            text = ''.join(text)
            text = text.replace('\n', '')
            # print(text)
            self.convert_string_to_integer(text, integer_sequence)

        return torch.IntTensor(integer_sequence), torch.IntTensor(lengths)

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        else:
            count = 0
            v = 0

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

