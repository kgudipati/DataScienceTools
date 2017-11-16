'''
K Nearest Neighbor Module
'''
import collections
import vectors as vec

# Take list of labels and return most common one
# assumes labels ordered neaarest to farthest
def majorityVote(labels):
    vote_ct = collections.Counter(labels)
    winner, winner_ct = vote_ct.most_common(1)[0]

    num_winners = len([count for count in vote_ct.values() if count == winner_ct])

    if num_winners == 1:
        # only one winning vote
        return winner
    else:
        # try again without the farthest, reduce k
        return majorityVote(labels[:-1])


# KNN Classifier
# k: # of neighbors to vote from
# train_data: list of tuples (data, labels)
# new_data: tuple of new data to classify
def classify(k, train_data, new_data):

    # order the training data from nearest to farthest
    ordered_data = sorted(train_data, key=lambda (train_data_vec, _): vec.distance(train_data_vec, new_data))

    # find the k-nearest labels
    k_nearest_labels = [label for _, label in ordered_data[:k]]

    # use the nearest labels to get majority label vote
    return majorityVote(k_nearest_labels)