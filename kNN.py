'''
K Nearest Neighbor Module
'''
import collections
import vectors as vec

class kNN():

    # ctor
    def __init__(self, train, labels):
        self.training_data = zip(train, labels) # list of tuple [(training vector, label)]
        self.train = train
        self.labels = labels


    # Take list of labels and return most common one
    # assumes labels ordered neaarest to farthest
    def majorityVote(self, labels):
        vote_ct = collections.Counter(labels)
        winner, winner_ct = vote_ct.most_common(1)[0]

        num_winners = len([count for count in vote_ct.values() if count == winner_ct])

        if num_winners == 1:
            # only one winning vote
            return winner
        else:
            # try again without the farthest, reduce k
            return self.majorityVote(labels[:-1])


    # Classify the given new data using k # of neighbors
    # k: # of neighbors to vote from
    # new_data: tuple of new data to classify
    def classify(self, k, new_data):

        # order the training data from nearest to farthest
        ordered_data = sorted(self.training_data, key=lambda (train_data_vec, _): vec.distance(train_data_vec, new_data))

        # find the k-nearest labels
        k_nearest_labels = [label for _, label in ordered_data[:k]]

        # use the nearest labels to get majority label vote
        return self.majorityVote(k_nearest_labels)