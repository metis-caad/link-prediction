import matplotlib.pyplot as plt

for plt_type in ['auc', 'train_loss']:
    if plt_type == 'auc':
        xlbl = 'Training steps'
        ylbl = 'Accuracy'
        clr = 'blue'
    else:
        xlbl = 'Training steps'
        ylbl = 'Loss'
        clr = 'orange'

    with open(plt_type + '.csv', 'r') as csv:
        lines = csv.readlines()
        x_axis = []
        y_axis = []
        for line in lines:
            data = line.split(',')
            x_axis.append(int((data[0])))
            y_axis.append(float(str(data[1])[:4]))
        linestyle = '-'
        plt.ylim(0, 1)
        plt.xlim(0, 500)
        plt.plot(x_axis, y_axis, linestyle=linestyle, color=clr, label=ylbl + ' @ Step', marker='o')

    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend()
    plt.savefig(plt_type + '.png', dpi=300)
    plt.show()
