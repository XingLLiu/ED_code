from ED_support_module import *


def predict_proba(self, x_data):
    '''
    Predict the outlier score of x_data.
    Input : x_data = [DataFrame or array] x test set
    '''
    try:
        # Get values if x_data is DataFrame
        x_data = x_data.values
    except:
        pass
    anomaly_scores = self.compute_paths(X_in = x_data)
    return anomaly_scores


def predict(self, x_data, outlier_proportion, anomaly_scores=None):
    '''
    Predict the response variable using x_data.
    Input : x_data = [DataFrame] design matrix. Omitted if anomaly_score
                                 is not None.
            outlier_proportion = [float] proportion of outliers required.
                                         Must be between 0 and 1.
            anomaly_scroes = [Series or array] anomaly scores of the
                             instances. The higher the score, the more
                             likely it is an outlier. If None, predict
                             by using x_data first.
    Output: y_pred = [Series] predicted response vector. 1 for outlier.
    '''
    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("Type of x_data must be DataFrame but got {} instead."
                        .format(type(x_data)))
    if anomaly_scores is None:
        anomaly_scores = self.predict_proba(x_data)
    # sort the scores
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    # retrieve indices of anomalous observations
    outlier_num = int( np.ceil( outlier_proportion * x_data.shape[0] ) )
    indices_with_preds = anomaly_scores_sorted[ -outlier_num : ]
    # create predictions
    y_pred = x_data.iloc[:, 0] * 0
    y_pred.iloc[indices_with_preds] = 1
    return y_pred


def plot_scores(self, anomaly_scores, y_true, y_pred, save_path=None, title=None, eps=False):
    '''
    Plot the anomaly scores.
    Input : anomaly_scores = [Series or array] anomaly scores.
            y_true = [Series or array] response vector.
            title = title of the plot
    '''
    # Convert to numpy array
    anomaly_scores = np.array(anomaly_scores)
    # Plot scores
    x_vec = np.linspace(1, len(y_true), len(y_pred))
    # True positives
    true_positive = (y_true == 1) & (y_pred == 1)
    _ = sns.scatterplot(x = x_vec, y = anomaly_scores)
    _ = sns.scatterplot(x = x_vec[y_true == 1],
                        y = anomaly_scores[y_true == 1],
                        label = "false negatives")
    _ = sns.scatterplot(x = x_vec[y_pred == 1],
                        y = anomaly_scores[y_pred == 1],
                        label = "false positives")
    _ = sns.scatterplot(x = x_vec[true_positive == 1],
                        y = anomaly_scores[true_positive == 1],
                        label = "true positives")
    _ = plt.title(title)
    _ = plt.legend()
    if save_path is not None:
        if eps:
            plt.savefig(save_path + "scores.eps", format="eps", dpi=800)
        else:
            plt.savefig(save_path + "scores.png")
        plt.close()
    else:
        plt.show()





eif.iForest.predict_proba = predict_proba
eif.iForest.predict = predict
eif.iForest.plot_scores = plot_scores

