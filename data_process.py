import os
import datetime

base_path = '/data/NLP/ieee_zhihu_cup'

train_set_path = os.path.join(base_path, 'question_train_set.txt')
relation_path = os.path.join(base_path, 'question_topic_train_set.txt')

valid_train_path = os.path.join(base_path, 'train_data')
valid_test_path = os.path.join(base_path, 'test_data')
output_train_path = os.path.join(base_path, 'train.txt')

interval = 10


def combine_to_line(train_list, relation_list):
    label_list = relation_list[1].split(',')
    label_list_add = ['__label__' + item for item in label_list]
    result = ' '.join(label_list_add) + '\t' + ' '.join(train_list[2].split(',')) + ' ' + ' '.join(
        train_list[4].split(','))
    return result


start = datetime.datetime.now()
with open(train_set_path, encoding='utf-8') as f_train_set:
    with open(relation_path, encoding='utf-8') as f_relation:
        with open(output_train_path, mode='w', encoding='utf-8') as fw_train:
            with open(valid_train_path, mode='w', encoding='utf-8') as fw_valid_train:
                with open(valid_test_path, mode='w', encoding='utf-8') as fw_valid_test:
                    i = 0
                    while True:
                        i += 1
                        train_line = f_train_set.readline()
                        if train_line == '':
                            break
                        train_list = train_line.split('\t')
                        relation_list = f_relation.readline().split()
                        line = combine_to_line(train_list, relation_list)
                        fw_train.write(line)
                        if i % interval == 0:
                            fw_valid_test.write(line)
                        else:
                            fw_valid_train.write(line)

now = datetime.datetime.now()
print('data process time is: {}'.format(now - start))
