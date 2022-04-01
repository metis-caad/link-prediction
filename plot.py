import matplotlib.pyplot as plt

for plt_type in ['auc', 'train_loss']:
    if plt_type == 'auc':
        xlbl = 'Training steps'
        ylbl = 'Accuracy'
        clr = 'dodgerblue'
        y_from = 0
        y_to = 1
    else:
        xlbl = 'Training steps'
        ylbl = 'Loss'
        clr = 'darkorange'
        y_from = 0.33
        y_to = 0.96

    with open(plt_type + '.csv', 'r') as csv:
        lines = csv.readlines()
        x_axis = []
        y_axis = []
        for line in lines:
            data = line.split(',')
            x_axis.append(int((data[0])))
            y_axis.append(float(str(data[1])[:4]))
        plt.ylim(y_from, y_to)
        plt.xlim(0, 500)
        plt.plot(x_axis,
                 y_axis,
                 linestyle='-',
                 linewidth=1.75,
                 color=clr,
                 label=ylbl + ' @ Step',
                 marker='s',
                 markevery=50,
                 markersize=12,
                 markeredgecolor='black',
                 aa=True,
                 snap=True)
        plt.annotate(str(y_axis[0]), xy=(x_axis[0], y_axis[0]), xytext=(10, 0), textcoords='offset points', size=12)
        plt.annotate(str(y_axis[len(y_axis) - 1]), xy=(x_axis[len(x_axis) - 1], y_axis[len(y_axis) - 1]),
                     xytext=(5, 0), textcoords='offset points', size=12)

    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend()
    plt.savefig(plt_type + '.png', dpi=300, bbox_inches='tight')
    plt.show()
