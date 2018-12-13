import pandas as pd
import numpy as np


# not implemented, goal would be to consider the top 3 votes
# and not only the top 1 goal like the current solution

def weighted_voting():
    for col in df.columns:
        vote = df[col].value_counts()  # count occurrences, result in ascending order
        n = df.columns.get_loc(col) + 1  # get next column, containing the likelihoods
        model = pd.to_numeric(
            df.iloc[:, n]).idxmax()  # get index of highest score (model_Name with highest precision)
        ensemble_votes.set_value(index=0, col=col, value=df[col][model])  # set vote to output with highest precision


# count majority of votes, by equality take model with highest confidence

def majority_voting():
    for col in df[df.columns[::2]]:  # only look at every 2nd column, since the others contain the likelihood
        vote = df[col].value_counts()  # count occurrences, result in ascending order
        if vote[0]>1:  # if count for one item is higher then one, add this item
            ensemble_votes.at[0, col] = vote.nlargest(1).index[0]  # get row name of first item
        else: # return model prediction with highest confidence
            n = df.columns.get_loc(col) + 1  # get index of next column, containing the likelihoods
            model = pd.to_numeric(df.iloc[:,n]).idxmax()  # get index of highest score (model_Name with highest precision)
            ensemble_votes.at[0, col] = df[col][model] # set vote to output with highest precision


# checks for all predictions if they match the true Label

def evaluation():
    for i in range(len(trueLabels)):    # number of labels
        for x in range (len(list(ensemble_votes.index))):    # number of voters: models + ensemble
            if trueLabels[i] == ensemble_votes.iat[x, i]:     # compare true label to prediction, if correct increase the counter
                evaluation_result[x] = evaluation_result[x]+1


# load df with this format:
# true Label:  house  | house_likelihood ; room  | room_likelihood
#    model 1:  apples | 0.2%             ; room  | 0.8%
#    model 2:  house  | 0.5%             ; room  | 0.7%
df = pd.read_pickle('../predictionMix_for: resnet50, vgg16, densenet')
trueLabels = df.columns[::2].values     # array containing only the class labels and not the likelihood
evaluation_result = np.zeros(len(df.index) + 1, dtype= int)     # array to store all model predictions and the ensemble prediction
ensemble_votes = pd.DataFrame(columns=df.columns[::2], dtype= str)       # dataframe to store ensembles vote

majority_voting()

df.drop(df.columns[1::2], inplace=True, axis=1)     # prepare df to merge it with ensemble_votes
ensemble_votes = ensemble_votes.append(df)      # prediction from all models and the ensemble
ensemble_votes.rename(index={0: 'ensemble'}, inplace=True)      # add axis name for ensemble row

evaluation()

with pd.option_context('expand_frame_repr', False):
    print(ensemble_votes, '\n')
    print('number of correct predictions for each model: ', evaluation_result)
    print('accuracy of each model: ', np.true_divide(evaluation_result,len(trueLabels)))





