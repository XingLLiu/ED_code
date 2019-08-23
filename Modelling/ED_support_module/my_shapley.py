import numpy as np
import pandas as pd
import sklearn as sk
from scipy.special import comb

from ED_support_module import *

def main():
    np.random.seed(4)

    gp1 = np.random.normal(0, 1, [25, 3])
    gp2 = np.random.normal(0, 1, [25, 3])
    gp3 = np.random.normal(0, 1, [25, 3])

    y_1 = ( 1 / ( 1 + np.exp( -gp1[:, 1] - 2 * gp1[:, 0] ) ) ) > 0.5
    y_2 = ( 1 / ( 1 + np.exp( -gp2[:, 1] - 2 * gp2[:, 0] ) ) ) > 0.5
    y_3 = ( 1 / ( 1 + np.exp( -gp3[:, 1] - 2 * gp3[:, 0] ) ) ) < 0.1

    x_train = pd.DataFrame( np.concatenate([gp1[:20, :], gp2[:20, :], gp3[:20, :]], axis = 0), columns = ["0", "1", "2"] )
    y_train = pd.Series( np.concatenate([y_1[:20], y_2[:20], y_3[:20]], axis = 0), name = "response", dtype = int )
    # x_train = pd.DataFrame( np.concatenate([gp2[:20, :], gp1[:20, :], gp3[:20, :]], axis = 0), columns = ["0", "1", "2"] )
    # y_train = pd.Series( np.concatenate([y_2[:20], y_1[:20], y_3[:20]], axis = 0), name = "response", dtype = int )
    train_data = pd.concat([x_train, y_train], axis = 1)
    train_dict = {"0":train_data.loc[:19, :], "1":train_data.loc[20:39, :], "2":train_data.loc[40:59, :]}



    x_test = pd.DataFrame( np.concatenate([gp1[20:, :], gp2[20:, :], gp3[20:, :]], axis = 0), columns = ["0", "1", "2"] )
    y_test = pd.Series( np.concatenate([y_1[20:], y_2[20:], y_3[20:]], axis = 0), name = "response", dtype = int )
    # x_test = pd.DataFrame( np.concatenate([gp2[20:, :], gp1[20:, :], gp3[20:, :]], axis = 0), columns = ["0", "1", "2"] )
    # y_test = pd.Series( np.concatenate([y_2[20:], y_1[20:], y_3[20:]], axis = 0), name = "response", dtype = int )
    test_data = pd.concat( [x_test, y_test], axis = 1 )

    benchmark_score = None

    # model = sk.linear_model.LogisticRegression(solver = "liblinear").fit(x_train, y_train)
    # model.coef_

    # model2 = sk.linear_model.LogisticRegression(solver = "liblinear").fit(x_train.loc[:39, :], y_train.loc[:39])
    # model2.coef_
    # (model2.predict(x_test) == y_test).sum()


    # pred_prob = model2.predict_proba(x_test)[:, 1]
    # y_pred = threshold_predict(pred_prob, y_test, 0.5)
    # true_positive_rate(y_test, y_pred)

    shapley_vec = shapley(model_class = sk.linear_model.LogisticRegression(solver = "liblinear"),
                        train_dict = train_dict,
                        test_data = test_data,
                        fpr_threshold = 0.5,
                        convergence_tol = 0.01,
                        performance_tol = 0.01,
                        max_iter = 200,
                        benchmark_score = None)

    for which in ["0", "1", "2"]:
        _ = sns.scatterplot(range(shapley_vec.shape[0]), shapley_vec.loc[:, which], label = which)


    _ = plt.legend()
    plt.show()


    # Computing Shapley using the explicity formula
    shapley_exact(model_class = sk.linear_model.LogisticRegression(solver = "liblinear"),
                     train_dict = train_dict,
                     test_data = test_data,
                     fpr_threshold = 0.5,
                     convergence_tol = 0.01,
                     performance_tol = 0.01,
                     max_iter = 200,
                     benchmark_score = None)




def shapley(model_class, train_dict, test_data, fpr_threshold,
            convergence_tol, performance_tol, max_iter, benchmark_score):
    '''
    iter_ind = t
    gp_ind = j
    '''
    groups = list( train_dict.keys() )
    added_groups = ["init"] + groups
    shapley_vec = pd.DataFrame(0, index = range(max_iter + 1), columns = groups)
    iter_ind = 0
    convergence_err = convergence_tol + 1
    scores = pd.DataFrame(0, index = range(max_iter + 1), columns = added_groups)
    # Test set
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # Compute benchmark score
    if benchmark_score is None:
        # Construct the entire train set
        first_key = groups[0]
        train_data = train_dict[first_key]
        for name in groups[1:]:
            train_data = pd.concat( [ train_data, train_dict[name] ], axis = 0 )
        # Get design matrix
        x_train = train_data.iloc[:, :-1]
        # Get labels
        y_train = train_data.iloc[:, -1]
        # Fit model and evaluate metric
        model = model_class.fit(x_train, y_train)
        pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred = threshold_predict(pred_prob, y_test, fpr_threshold)
        benchmark_score = true_positive_rate(y_test, y_pred)
    # MC shapley
    while (convergence_err > convergence_tol) and (iter_ind < max_iter):
        print(iter_ind)
        iter_ind += 1
        # Permute the groups
        perm = np.random.permutation(groups)
        # Add a dummy index for convenience
        added_perm = ["init"] + list(perm)
        for gp_ind in range(1, len(added_groups)):
            # Current group name
            gp_name = added_perm[gp_ind]
            # Early stopping if marginal improvement is small
            if abs(benchmark_score - scores.loc[iter_ind, gp_name]) < performance_tol:
                # Note that a new column of zeros has been added to scores
                prev_gp_name = added_perm[gp_ind - 1]
                scores.loc[iter_ind, gp_name] = scores.loc[iter_ind, prev_gp_name]
            else:
                # Retrieve train data upto pi(j) as in the paper
                train_data = train_dict[added_perm[1]]
                for name in added_perm[2:gp_ind + 1]:
                    train_data = pd.concat( [ train_data, train_dict[name] ], axis = 0 )
                # Get design matrix
                x_train = train_data.iloc[:, :-1]
                # Get labels
                y_train = train_data.iloc[:, -1]
                # Fit model and evaluate metric
                model = model_class.fit(x_train, y_train)
                pred_prob = model.predict_proba(x_test)[:, 1]
                y_pred = threshold_predict(pred_prob, y_test, fpr_threshold)
                scores.loc[iter_ind, gp_name] = true_positive_rate(y_test, y_pred)
            # Update data shapley
            prev_gp_name = added_perm[gp_ind - 1]
            shapley_vec.loc[iter_ind, gp_name] = ( (iter_ind - 1) / iter_ind ) * shapley_vec.loc[iter_ind - 1, gp_name] + ( scores.loc[iter_ind, gp_name] - scores.loc[iter_ind, prev_gp_name] ) / iter_ind
        if iter_ind == max_iter - 1:
            print("Warning: Maximum iteration of {} reached before convergence".format(max_iter))
            return shapley_vec
    return shapley_vec







def shapley_exact(model_class, train_dict, test_data, fpr_threshold,
                convergence_tol, performance_tol, max_iter, benchmark_score,
                model_name, num_epochs, batch_size, optimizer, criterion, device):
    groups = list( train_dict.keys() )
    power_set = list_powerset(groups)
    power_set.remove([])
    # Initialize shapley
    shapley_vec = pd.DataFrame(0, index = range(1), columns = groups)
    # Separate test data
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    for current_gp in groups:
        print("Computing Shapley for {}".format(current_gp))
        shapley = 0
        for subgp in power_set:
            if current_gp not in subgp:
                input_size = x_test.shape[0]
                DROP_PROB = 0.4
                HIDDEN_SIZE = 500
                BATCH_SIZE = 128
                NUM_EPOCHS = 100
                LEARNING_RATE = 1e-3
                CLASS_WEIGHT = 3000
                model_class = NeuralNet(device = device,
                                        input_size = input_size,
                                        drop_prob = DROP_PROB,
                                        hidden_size = HIDDEN_SIZE).to(device)
                criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
                optimizer = torch.optim.SGD(model_class.parameters(), lr = LEARNING_RATE)
                summand = shapley_summand(model_class, subgp, current_gp, train_dict,
                                            x_test, y_test, fpr_threshold,
                                            model_name, num_epochs, batch_size, optimizer,
                                            criterion, device)
                shapley += summand
        shapley_vec[current_gp] = shapley
    return shapley_vec



def shapley_summand(model_class, subgp, current_gp, train_dict, x_test, y_test, fpr_threshold,
                    model_name, num_epochs, batch_size, optimizer, criterion, device):
    '''
    Compute the summand.
    '''
    # Retrieve train data upto pi(j) as in the paper
    train_data = train_dict[subgp[0]]
    if len(subgp) > 1:
        for name in subgp[1:]:
            train_data = pd.concat( [ train_data, train_dict[name] ], axis = 0 )
    # S union i
    train_data_large = pd.concat( [ train_data, train_dict[current_gp] ], axis = 0 )
    # Evaluate metrics
    scores = [0, 0]
    for k, data in enumerate([train_data, train_data_large]):
        # Get design matrix
        x_train = data.iloc[:, :-1]
        # Get labels
        y_train = data.iloc[:, -1]
        # Fit model and evaluate metric
        if model_name == "logistic":
            model = model_class.fit(x_train, y_train)
            pred_prob = model.predict_proba(x_test)[:, 1]
        elif model_name == "nn":
            model = model_class.fit_model(self = model_class, x_data = x_train,
                                    y_data = y_train,
                                    num_epochs = num_epochs,
                                    batch_size = batch_size,
                                    optimizer = optimizer,
                                    criterion = criterion)
            pred_prob = model.predict_proba(x_test)[:, 1]
        # y_pred = pred_prob > 0.5
        # scores[k] = sk.metrics.accuracy_score(y_test, y_pred)
        y_pred = threshold_predict(pred_prob, y_test, fpr_threshold)
        scores[k] = true_positive_rate(y_test, y_pred)
    
    return (scores[1] - scores[0]) / comb( len( train_dict ) - 1, len( subgp ) )



def list_powerset(lst):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in lst:
        result.extend([subset + [x] for subset in result])
    return result
 