from ED_support_module import *

class Evaluation:
    def __init__(self, y_test, pred_prob):
        self.pred_prob = pred_prob
        self.y_test = y_test
    def roc_plot(self, plot=True, title="", n_pts = 51, save_path = None):
        '''
        Plot the roc curve of a trained logistic regression model.
        Input:  self.y_test = test set (pd.dataframe or series)
        Output: ROC plot
        '''
        plt.close()
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
        plt.title(f'Receiver Operating Characteristic ({title})')
        plt.plot(fpr_lst, tpr_lst, 'b', label = 'AUC = %0.3f' % roc_auc)
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
                plt.close()
            else:
                plt.savefig(save_path)
                plt.close()
        if plot:
            plt.show()
        return( pd.DataFrame( {'TPR':tpr_lst, 'FPR':fpr_lst} ) )
    def summary(self):
        '''
        Return a DataFrame with TPR, FPR, TP, FN, FP, TN.
        '''
        # Compute TPR and FPR
        summary = self.roc_plot(plot=False)
        # No. of positives
        p_num = self.y_test.sum()
        # No. of negatives
        n_num = len(self.y_test) - p_num
        # Compute no. of TP and FN
        summary['TP'] = (summary['TPR'] * p_num).round().astype('int')
        summary['FN'] = p_num - summary['TP']
        # Compute no. of FP and TN
        summary['FP'] = (summary['FPR'] * n_num).round().astype('int')
        summary['TN'] = n_num - summary['FP']
        return(summary)
    def roc_subplot(self, data_path, save_path, time_span, dim, eps=False):
        '''
        ROC subplot of all months.
        Input : data_path = [str] Path to the folder containing subfolders of data of 
                            each month. Data must contain "TPR" and "FPR"
                save_path = [str] path to save figure. Equals data_path by default.
                time_span = [list of int] time span of the data in the form YYYYMM.
                dim = [list] list of 2 integers indicating the dimension of
                      subplot.
        '''
        for i, time in enumerate(time_span[3:]):
            # Load data
            try:
                csv_name = data_path + f'{time}/summary_{time}.csv'
                summary = pd.read_csv(csv_name)
                print("{}".format(time))
            except:
                Warning(f"Error in loading data for {time}! Check if it exists or contains columns \'TPR\', \'FPR\'.")
                continue
            _ = plt.subplot(dim[0], dim[1], i + 1)
            # ROC plot
            tpr = summary['TPR']
            fpr = summary['FPR']
            roc_auc = sk.metrics.auc(fpr, tpr)
            month_pred = time_span[i + 3]
            _ = plt.title(f'ROC {month_pred}')
            _ = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            _ = plt.legend(loc = 'lower right')
            _ = plt.plot([0, 1], [0, 1],'r--')
            _ = plt.xlim([0, 1])
            _ = plt.ylim([0, 1])
            _ = plt.ylabel('True Positive Rate')
            _ = plt.xlabel('False Positive Rate')
        plt.tight_layout()
        if eps:
            plt.savefig(save_path + 'aggregate_roc.eps', format='eps', dpi=1000)
        else:
            plt.savefig(save_path + 'aggregate_roc.png')
        plt.close()


