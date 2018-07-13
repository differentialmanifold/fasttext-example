import os
import datetime
import fasttext

train_path = '/data/NLP/ieee_zhihu_cup/train_data'
valid_path = '/data/NLP/ieee_zhihu_cup/test_data'
model_path = '/data/NLP/ieee_zhihu_cup/model/model_zh'


class MyClassify:
    def __init__(self, model_path):
        self.model_path = model_path
        self.classifier = None
        model_parent_path = os.path.abspath(os.path.join(model_path, os.path.pardir))
        if not os.path.exists(model_parent_path):
            os.makedirs(model_parent_path)

    def train(self, train_path):
        start = datetime.datetime.now()

        self.classifier = fasttext.supervised(train_path, self.model_path, loss='hs')
        now = datetime.datetime.now()
        print('train finished, takes {}'.format(now - start))

    def test(self, valid_path):
        if self.classifier is not None:
            result = self.classifier.test(valid_path)
            print(result.precision)
            print(result.recall)
            print(result.nexamples)

    def load_model(self):
        self.classifier = fasttext.load_model(self.model_path + '.bin')


if __name__ == '__main__':
    my_classify = MyClassify(model_path)

    my_classify.train(train_path)

    my_classify.test(valid_path)
