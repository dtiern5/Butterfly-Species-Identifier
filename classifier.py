import matplotlib.pyplot as plt

from labels import labels, classification


def machine_classification(prediction):
    ans = prediction.argmax(axis=-1)
    y = " ".join(str(x) for x in ans)
    y = int(y)
    return labels[y]


def get_percentages(prediction):
    # Get labels in order so we can put a percentage with the prediction
    list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = prediction

    # Simple swapping function to sort the predictions least -> greatest
    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    # top 3 predictions
    percentages = []
    for i in range(10):
        percentages.append("{}: {}%".format(classification[list_index[i]],round(prediction[0][list_index[i]] * 100, 2)))

    return percentages


def bar_graph_predictions(predictions):
    list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = predictions

    # Simple swapping function to sort the predictions least -> greatest
    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    # labels in order
    print(list_index)

    percentages = []
    for i in range(10):
        percentages.append(
            "{}: {}%".format(classification[list_index[i]], round(predictions[0][list_index[i]] * 100, 2)))

    percentage_only = []
    for i in range(10):
        percent = round(predictions[0][list_index[i]] * 100, 2)
        percentage_only.append(percent)

    fig, ax = plt.subplots()
    fig.set_size_inches(22, 14)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    plt.rc('font', **font)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.ylim(0, 100)

    ax.bar(percentages, percentage_only, color='purple', edgecolor='green')

    return fig