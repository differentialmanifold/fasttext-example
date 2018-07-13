import math
import fasttext

model_path = '/data/NLP/ieee_zhihu_cup/model/model_zh'
full_model_path = '/data/NLP/ieee_zhihu_cup/model/model_zh_full'
full_train_path = '/data/NLP/ieee_zhihu_cup/train.txt'
test_path = '/data/NLP/ieee_zhihu_cup/test_data'
eval_set_path = '/data/NLP/ieee_zhihu_cup/question_eval_set.txt'
eval_result_path = '/data/NLP/ieee_zhihu_cup/eval_result.csv'


def process_test_data():
    processed_list = []
    classify = fasttext.load_model(model_path + '.bin')
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            labels, contents = line.split('\t')
            predicted_contents = classify.predict([contents], k=5)[0]
            processed_list.append((predicted_contents, labels.split()))
    return processed_list


def eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall)


def generate_result():
    classify = fasttext.load_model(full_model_path + '.bin')
    with open(eval_set_path, encoding='utf-8') as f:
        with open(eval_result_path, mode='w', encoding='utf-8') as fw:
            for line in f:
                eval_list = line.split('\t')
                eval_line = ' '.join(eval_list[2].split(',')) + ' ' + ' '.join(eval_list[4].split(','))
                eval_labels = classify.predict([eval_line], k=5)[0]
                eval_labels = [item[len('__label__'):] for item in eval_labels]
                fw.write(eval_list[0] + ',' + ','.join(eval_labels) + '\n')


if __name__ == '__main__':
    generate_result()
