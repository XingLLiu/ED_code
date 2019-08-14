class Person:
    def __init__(self, name, age, maths):
        self.name = name
        self.age = age
        self.maths = maths
    def greeting(self):
        print("Hello, my name is {}.".format(self.name))
    def favorite_maths(self):
        print("My favorite maths is {}.".format(self.maths))



# ----------------------------------------------------
from ED_support_module import *
import argparse

timeSpan = [201807, 201808, 201809, 201810, 201811, 201812, 201901, 201902,
            201903, 201904, 201905, 201906]

# Create a directory if not exists
plot_path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/neural_net/'
dynamic_plot_path = plot_path + 'dynamic/'

# Create subplot
for i, month in enumerate(timeSpan[1:-2]):
    csv_name = dynamic_plot_path + f'summary_{month}.csv'
    summary = pd.read_csv(csv_name)
    _ = plt.subplot(3, 3, i + 1)
    # ROC plot
    tpr = summary['TPR']
    fpr = summary['FPR']
    roc_auc = sk.metrics.auc(fpr, tpr)
    month_pred = timeSpan[i + 3]
    _ = plt.title(f'ROC {month_pred}')
    _ = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    _ = plt.legend(loc = 'lower right')
    _ = plt.plot([0, 1], [0, 1],'r--')
    _ = plt.xlim([0, 1])
    _ = plt.ylim([0, 1])
    _ = plt.ylabel('True Positive Rate')
    _ = plt.xlabel('False Positive Rate')


plt.tight_layout()
plt.savefig(dynamic_plot_path + 'aggregate_roc.eps', format='eps', dpi=1000)
plt.show()


for i, month in enumerate(timeSpan[1:-2]):
    csv_name = dynamic_plot_path + f'summary_{month}.csv'
    if i == 0:
        summary = pd.read_csv(csv_name)
    else:
        summary += pd.read_csv(csv_name)


# ROC plot
tpr = summary['TP'] / (summary['TP'] + summary['FN'])
fpr = summary['FP'] / (summary['TN'] + summary['FP'])
roc_auc = sk.metrics.auc(fpr, tpr)
month_pred = timeSpan[i + 2]
_ = plt.title('One-month Ahead Aggregate ROC')
_ = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
_ = plt.legend(loc = 'lower right')
_ = plt.plot([0, 1], [0, 1],'r--')
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1])
_ = plt.ylabel('True Positive Rate')
_ = plt.xlabel('False Positive Rate')
plt.savefig(dynamic_plot_path + 'aggregate_roc.eps', format='eps', dpi=1000)
plt.show()


# ----------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default=None,
                        type=str,
                        required=True,
                        help="The mode to be used.")
    return parser
    

parser = setup_parser()
args = parser.parse_args()

print(args.mode)