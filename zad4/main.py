import random
from typing import Optional

import pandas as pd
import numpy as np


dataset = "data/car.data"
ATTRIBUTES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
Y_ATTRIBUTE = 'class'
Y_CLASSES = ['unacc', 'acc', 'good', 'vgood']

# dataset = "data/example.data"
# ATTRIBUTES = ['x1', 'x2']
# Y_ATTRIBUTE = 'y'
# Y_CLASSES = ['0', '1']


class Node:
    def __init__(self):
        self.children = {}
        self.value = ''
        self.is_leaf = False
        self.predictions = ''


def hot_encode_data(data, attributes: list[str]):
    return pd.get_dummies(data, attributes)


def read_data(attributes: list[str]):
    data = pd.read_csv(dataset, index_col=False)
    # hot_encode_data(data, categories)
    return data


def entropy(df) -> float:
    target = df[Y_ATTRIBUTE]

    entropy_value = 0.0
    unique_values, numbers = np.unique(target, return_counts=True)

    for i in range(len(unique_values)):
        fraction = numbers[i] / sum(numbers)
        entropy_value += -fraction * np.log2(fraction)

    return entropy_value


def info_gain(df, target_attribute_name: str) -> float:
    target_attribute = df[target_attribute_name]

    gain_value = entropy(df)
    unique_attribute_values, attribute_numbers = np.unique(target_attribute, return_counts=True)

    for i in range(len(unique_attribute_values)):
        d = df.where(df[target_attribute_name] == unique_attribute_values[i])
        d = d.dropna()
        entropy_value = entropy(d)

        gain_value -= (float(len(d)) / float(len(df))) * entropy_value

    return gain_value


def find_most_informative_attribute(df, attributes) -> Optional[str]:
    max_info_gain = 0
    best_attribute = None

    for attribute in attributes:
        gain = info_gain(df, attribute)
        if gain > max_info_gain:
            max_info_gain = gain
            best_attribute = attribute

    return best_attribute


def id3(df, attributes):
    node = Node()

    best_attribute = find_most_informative_attribute(df, attributes)
    node.value = best_attribute

    if best_attribute is None:
        return node

    unique_values = np.unique(df[best_attribute])
    for unique_value in unique_values:
        sub_df = df.where(df[best_attribute] == unique_value)
        sub_df = sub_df.dropna()

        if entropy(sub_df) == 0.0:
            new_node = Node()
            new_node.is_leaf = True

            new_node.predictions = np.unique(sub_df[Y_ATTRIBUTE])[0]
            node.children[str(unique_value)] = new_node
        else:
            new_attributes = attributes.copy()
            new_attributes.remove(best_attribute)
            child = id3(sub_df, new_attributes)
            node.children[str(unique_value)] = child

    return node


def print_tree(root_node: Node, depth: int = 0):
    print('\t' * depth, end='')
    print(root_node.value, end='')
    if root_node.is_leaf:
        print(root_node.predictions)
    print()
    for key, child in root_node.children.items():
        print('\t' * depth, end='')
        print(f'Key: {key}')
        print_tree(child, depth+1)


def predict(node: Node, row):
    if node.is_leaf:
        return node.predictions
    else:
        if node.value is None:
            print(f'error!')
            return

        value = str(row[node.value])

        if value is None:
            print(f'error! {node.value}')
            return

        for key, child in node.children.items():
            if key == value:
                return predict(child, row)
        else:
            return random.choice(Y_CLASSES)


def calculate_measures(confusion_matrix):
    measures = {}

    for index, y_class in enumerate(Y_CLASSES):
        tp = confusion_matrix[index, index]
        fn = sum(confusion_matrix[index]) - tp
        fp = sum(confusion_matrix[:, index]) - tp
        tn = np.sum(confusion_matrix) - fn - fp - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        measures[y_class] = {
            # 'tp': tp,
            # 'fn': fn,
            # 'fp': fp,
            # 'tn': tn,
            'recall': recall,
            'fallout': fp / (fp + tn),
            'precision': precision,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1-score': (2 * precision * recall) / (recall + precision),
        }

    return measures


def shuffle_data(data):
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def split_data(data, k: int, index: int = 0):
    assert index < k

    rows_in_split = len(data) // k

    start_test = index * rows_in_split
    end_test = start_test + rows_in_split

    train_data_1 = data.iloc[:start_test]
    test_data = data.iloc[start_test:end_test]
    train_data_2 = data.iloc[end_test:]

    train_data = pd.concat([train_data_1, train_data_2])

    return train_data, test_data


def main():
    data = read_data(ATTRIBUTES)
    data = shuffle_data(data)

    train_data, test_data = split_data(data, 4, 0)

    root_node = id3(train_data, ATTRIBUTES)
    # print_tree(root_node)

    confusion_matrix = np.zeros((len(Y_CLASSES), len(Y_CLASSES)))

    for index, row in test_data.iterrows():
        row_as_dict = {attribute: row[attribute] for attribute in ATTRIBUTES}
        prediction = predict(root_node, row_as_dict)
        ground_truth = row[Y_ATTRIBUTE]

        if prediction is not None:
            confusion_matrix[Y_CLASSES.index(prediction)][Y_CLASSES.index(ground_truth)] += 1
        else:
            print('No prediction')

    measures = calculate_measures(confusion_matrix)
    for measure in measures:
        print(measure, measures[measure])

    df_confusion_matrix = pd.DataFrame(data=confusion_matrix, columns=Y_CLASSES, index=Y_CLASSES)
    df_test = pd.DataFrame(data=measures).transpose()

    print(confusion_matrix)


if __name__ == "__main__":
    main()
