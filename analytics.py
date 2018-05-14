import matplotlib.pyplot as plt
import numpy as np

"""
    This file will generate learning curve from the keras log file obtained from training a model
"""

if __name__ == "__main__":

    print("============== analytics.py ==============")

    #  Select one of the following
    if 0:
        filename = 'logs/custom_detector_log.csv'
        title = 'Custom Model'
    if 0:
        filename = 'logs/custom_classifier_log.csv'
        title = 'Custom Model'
    if 0:
        filename = 'logs/vgg_scratch_log.csv'
        title = 'VGG (Scratch) Model'
    if 1:
        filename = 'logs/vgg_pretrained_log.csv'
        title = 'VGG (Pretrained)'

    #  Read and convert the keras log file into a numpy array
    log = np.genfromtxt(filename, delimiter=',')
    epoch_num = list()
    for i in range(int(log[1:,0][-1]+1)):
        epoch_num.append(i)
    train_accuracy = log[1:, 1]*100
    train_loss = log[1:, 2]
    test_accuracy = log[1:, 3]*100
    test_loss = log[1:, 4]

    fig, ax1 = plt.subplots()
    x = epoch_num

    #  Set one of y axis to be testing accuracy
    y1 = test_accuracy
    ax1.plot(x, y1, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy (%)', color='b')
    ax1.tick_params('y', colors='b')

    #  Set other y axis to be testing loss
    ax2 = ax1.twinx()
    y2 = test_loss
    ax2.plot(x, y2, 'r-')
    ax2.set_ylabel('testing loss', color='r')
    ax2.tick_params('y', colors='r')

    #  Plot formatting
    fig.tight_layout()
    plt.xticks(epoch_num)
    plt.rc('grid', linestyle="-", color='black')
    ax1.grid(True)
    ax1.set_title('Learning Curve: ' + title)
    fig.set_size_inches(9, 5)
    plt.tight_layout()

    plt.show()
