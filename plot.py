from argparse import ArgumentParser

import matplotlib.pyplot as plt

# figure = plt.figure(figsize=(1000, 500)) # computer is hanging, bug?

with open('train_loss.csv', 'r') as csv:
    lines = csv.readlines()
    x_axis = []
    y_axis = []
    for line in lines:
        data = line.split(',')
        x_axis.append(data[0])
        y_axis.append(data[1])
    linestyle = '-'
    plt.plot(x_axis, y_axis, linestyle=linestyle, color='red', label='label', marker='o')
    # for a, b in zip(x_axis, y_axis):
    #     # if x_axis.index(a) == 4:
    #     plt.text(a, float(b), str(b),
    #              ha='center', va='center', color='white',
    #              bbox=dict(boxstyle='square,pad=0.15', fc=colors[label], ec='none'))

plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.legend()
plt.show()
# plt.savefig('ecaade2021_eval_' + eval_type + '.png', dpi=300)
