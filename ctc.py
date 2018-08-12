def CTCDecoder(class):

    def __init__(alphabel):
        self.alphabet = {' ': 0, 'a' : 1, 'b' : 2, 'c' : 3, 'd':  4,
                     'e': 5, 'f': 6, 'g': 7, 'h':8, 'i':9, 'j': 10, 'k': 11,
                     'l': 12, 'm' : 13, 'n' : 14, 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20,
                     'u' : 21, 'v' : 22, 'w' : 23, 'x' : 24, 'y' : 25, 'z' : 26, ',': 27, "'" : 28,
                     '': 29}

    def eval_forward_prob(self, output_timeseries, label):
        """ Finds the CTC score for the string label given the output distributions
            for all timesteps.

            output_timeseries       - T x D numpy array, where T
                                      is the length of timeseries of character distributions,
                                      and D is the size of the alphabet with the blank character
            label                   - a string
        """

        T = output_timeseries.shape[0]
        aug_label = self.preprocess_label(label)
        L = len(aug_label)

        # Initial probabilities
        # notation from the paper: alpha_t(s) = alpha[t, s]
        alpha_matrix = np.zeros(shape = (T, L))
        alpha_matrix[0, 0] = output_timeseries[0][aug_label[0]]
        alpha_matrix[0, 1] = output_timeseries[0][aug_label[1]]
        # ....
        # for all s > 1, alpha_matrix[0, s] = 0
        for t, char_dist in enumerate(output_timeseries[1:], 1):

            s = 0
            # finding the probability of prefix aug_label[0:s] at timestep t
            while (s =< t and s < L):
                # probability that current character was already reached in previous state
                reached = alpha_matrix[t - 1][aug_label[s]]
                # probability of transitioning from previous character (blank or same character) to the current
                prev_blank_same = alpha_matrix[t - 1][aug_label[s - 1]] if s >= 1 else 0
                alpha_hat = reached + prev_blank_same
                # adding probability of transitioning from previous distinct non-blank character to current one
                prev_distinct = alpha_matrix[t - 1][aug_label[s - 2]] if s >= 2 else 0
                #  (repeated characters => need blank) or (current character is blank)
                if (aug_label[s - 2] == aug_label[s] or aug_label[s] == self.alphabet['']):
                    alpha_matrix[t][s] = output_timeseries[t][aug_label[s]] * alpha_hat
                # previous character is a blank between two unique characters
                else:
                    alpha_matrix[t][s] = output_timeseries[t][aug_label[s]] * (alpha_hat + prev_distinct)
                s += 1
            # normalize the alphas for current timestep so that we don't underflow
            alpha_matrix = rescale_alpha(alpha_matrix, t, s + 1)
        return alpha_matrix[T - 1][L - 1] + alpha_matrix[T - 1][L - 2]


    def preprocess_label(self, label):
        """ Converts the labels to a sequence of character codes with
            a blank character between the original word's characters

            label                   - a string
        """
        aug_label = [].append(self.alphabel[''])
        for char in label:
            aug_label.append(self.alphabel[char])
            aug_label.append(self.alphabel[''])
        return aug_label

    #def predict_best_path(self, output_timeseries):

def rescale_alpha(alpha_matrix, t, s):
    """ Normalize probabilities for all (possible) prefixes of aug_label at timestep t

        alpha_matrix                -
        t                           -
        s                           -
    """
    alphas_sum = np.sum(alpha_matrix[t][:s])
    alpha_matrix[t][:s] /= alphas_sum
    return alpha_matrix


def test_eval_forward_prob():
    dec = CTCDecoder()
    label = "cat"
    output_timeseries_1 = np.array()
    output_timeseries_2 = np.array()
    assert 0 == dec.eval_forward_prob(output_timeseries_1, "cat")
    assert 0 == dec.eval_forward_prob(output_timeseries_2, "cat")
