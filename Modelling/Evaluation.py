from ED_support_module import *

class Evaluation:
    def __init__(self, y_test, pred_prob):
        self.pred_prob = pred_prob
        self.y_test = y_test

    def lr_roc_plot(self, plot=True, title="", n_pts = 51, save_path = None):
        '''
        Plot the roc curve of a trained logistic regression model.
        Input:  self.y_test = test set (pd.dataframe or series)
        Output: ROC plot
        '''
        fpr_lst, tpr_lst = [], []
        threshold = np.linspace(0, 1, n_pts)
        score_sorted = np.argsort(self.pred_prob)
        for i in range(n_pts):
            indices_with_preds = score_sorted[-int(np.ceil( threshold[i] * self.y_test.shape[0] )):]
            pred = self.y_test * 0
            pred.iloc[indices_with_preds] = 1
            fpr, tpr, _ = sk.metrics.roc_curve(self.y_test, pred)
            fpr_lst.append(fpr[1])
            tpr_lst.append(tpr[1])
        fpr_lst[-1], tpr_lst[-1] = 1, 1
        fpr_lst[0], tpr_lst[0] = 0, 0
        roc_auc = sk.metrics.auc(fpr_lst, tpr_lst)
        plt.title('Receiver Operating Characteristic ' + title)
        plt.plot(fpr_lst, tpr_lst, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # Save figure if required
        if save_path is not None:
            if ".eps" in save_path[-4:]:
                plt.savefig(save_path, format='eps', dpi=1000)
            else:
                plt.savefig(save_path)
            plt.close()
        if plot:
            plt.show()
        return({'TPR':tpr_lst, 'FPR':fpr_lst})


