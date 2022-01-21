import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = "data/iris.data"
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
output_class_attribute = "class"
output_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def read_data():
    return pd.read_csv(dataset, index_col=False)


def find_mean(series: pd.Series):
    # return np.average(series.to_numpy())
    return np.mean(series.to_numpy())


def find_standard_deviation(series: pd.Series):
    return np.std(series.to_numpy())


def find_gauss(x: float, mean, std):
    return (
        1
        / ((2.0 * math.pi * (std ** 2)) ** 0.5)
        * math.exp(-(((x - mean) ** 2) / (2.0 * (std ** 2))))
    )


def split_data(data, p=0.8):
    index = int(len(data) * p)
    train_data = data.iloc[:index]
    test_data = data.iloc[index:]

    return train_data, test_data


def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)


def float_decimal(value):
    try:
        v = int(value * 10000)
        return float(v) / 10000
    except:
        return "-"


def measure_dict(tp: int, fn: int, fp: int, tn: int) -> dict:
    precision = 0
    if tp + fp:
        precision = tp / (tp + fp)

    recall = 0
    if tp + fn:
        recall = tp / (tp + fn)

    fallout = 0
    if fp + tn:
        fallout = fp / (fp + tn)

    accuracy = 0
    if tp + tn + fp + fn:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    f1_score = 0
    if recall + precision:
        f1_score = (2 * precision * recall) / (recall + precision)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": float_decimal(recall),
        "fallout": float_decimal(fallout),
        "precision": float_decimal(precision),
        "accuracy": float_decimal(accuracy),
        "f1-score": float_decimal(f1_score),
    }


def calculate_measures(confusion_matrix):
    measures = {}

    for index, y_class in enumerate(output_classes):
        tp = confusion_matrix[index, index]
        fn = sum(confusion_matrix[index]) - tp
        fp = sum(confusion_matrix[:, index]) - tp
        tn = np.sum(confusion_matrix) - fn - fp - tp

        measures[y_class] = measure_dict(tp, fn, fp, tn)

    return measures


def calculate_measures_all(measures: dict):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for attribute, measure in measures.items():
        tp += measure["tp"]
        fn += measure["fn"]
        fp += measure["fp"]
        tn += measure["tn"]

    return measure_dict(tp, fn, fp, tn)


def predict(train_data, y):
    values = {}
    for output_class in output_classes:
        filtered_data = train_data.where(train_data[output_class_attribute] == output_class)
        filtered_data = filtered_data.dropna()

        class_probability = (len(filtered_data) + 1) / len(train_data)
        probabilities = [class_probability]

        for attribute in attributes:
            attribute_data = filtered_data[attribute]
            mean = find_mean(attribute_data)
            standard_deviation = find_standard_deviation(attribute_data)
            gauss = find_gauss(y[attribute], mean, standard_deviation)
            probabilities.append(gauss)

        normalized_probabilities = filter(lambda p: math.log(p), probabilities)
        values[output_class] = sum(normalized_probabilities)

    best_attribute = max(values, key=values.get)
    return best_attribute


def plot_attributes_with_class(data):
    for c in output_classes:
        filtered_data = data.where(data[output_class_attribute] == c)
        filtered_data = filtered_data.dropna()

        d1 = filtered_data["petal_length"].to_numpy()
        d2 = filtered_data["petal_width"].to_numpy()
        plt.scatter(d1, d2, label=c)

    plt.xlabel("petal_length")
    plt.ylabel("petal_width")

    plt.legend()
    plt.show()


def main():
    data = read_data()
    data = shuffle_data(data)

    # plot_attributes_with_class(data)

    train_data, test_data = split_data(data, p=0.3)

    confusion_matrix = np.zeros((len(output_classes), len(output_classes)))

    for index, row in test_data.iterrows():
        prediction = predict(train_data, row)
        ground_truth = row[output_class_attribute]
        confusion_matrix[output_classes.index(prediction)][output_classes.index(ground_truth)] += 1

    measures = calculate_measures(confusion_matrix)
    df_measures = pd.DataFrame(data=measures).transpose()

    measures_all = calculate_measures_all(measures)
    del measures_all['tn']
    del measures_all['tp']
    del measures_all['fn']
    del measures_all['fp']
    df_measures_all = pd.DataFrame(data={'value': measures_all}).transpose()


def draw_example_gauss(data):
    y = data["sepal_length"]
    mean = find_mean(y)
    std = find_standard_deviation(y)

    x = []
    i = 0.0
    while i < 10.0:
        g = find_gauss(i, mean, std)
        i += 0.05
        x.append(g)

    plt.plot(x)
    plt.show()


if __name__ == "__main__":
    main()
