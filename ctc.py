import numpy as np
import math

class CTCDecoder():

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def eval_forward_prob(self, output_timeseries, label):
        """ Inefficiently finds the CTC score for the string label given the RNN output distributions
            for all timesteps.

            output_timeseries       - T x D numpy array, where T
                                      is the length of timeseries of character distributions,
                                      and D is the size of the alphabet with the blank character
            label                   - a string
        """

        T = output_timeseries.shape[0]
        aug_label = self.preprocess_label(label)
        L = len(aug_label)

        # Converting to logprobs, so that we don't underflow
        output_timeseries = np.log(output_timeseries)

        # Initial probabilities
        # notation from the paper: alpha_t(s) = alpha[t, s]
        alpha_matrix = np.zeros(shape = (T, L))
        alpha_matrix[0, 0] = output_timeseries[0, aug_label[0]]
        alpha_matrix[0, 1] = output_timeseries[0, aug_label[1]]
        # ....
        # for all s > 1, alpha_matrix[0, s] = 0
        for t, char_dist in enumerate(output_timeseries[1:], 1):
            s = 0
            while (s < L):
                # probability that current character was already reached in previous timesteps
                reached = alpha_matrix[t - 1, s]
                # probability of transitioning from previous character (blank or same character) to the current
                prev_blank_same = alpha_matrix[t - 1, s - 1] if s >= 1 else 0

                alpha_hat = log_of_sum(reached, prev_blank_same)
                # adding probability of transitioning from previous distinct non-blank character to current one
                prev_distinct = alpha_matrix[t - 1, s - 2] if s >= 2 else 0
                #  (repeated characters => need blank) or (current character is blank)
                if (aug_label[s - 2] == aug_label[s] or aug_label[s] == self.alphabet['']):
                    alpha_matrix[t, s] = prod_of_logs(output_timeseries[t, aug_label[s]], alpha_hat)
                # previous character is a blank between two unique characters
                else:
                    alpha_matrix[t, s] = prod_of_logs(output_timeseries[t, aug_label[s]], log_of_sum(alpha_hat, prev_distinct))
                s += 1
            # normalize the alphas for current timestep so that we don't underflow
        return np.exp(alpha_matrix[T - 1, L - 1]) + np.exp(alpha_matrix[T - 1, L - 2])



    def preprocess_label(self, label):
        """ Converts the labels to a sequence of character codes with
            a blank character between the original word's characters

            label                   - a string
        """
        aug_label = []
        aug_label.append(self.alphabet[''])
        for char in label:
            aug_label.append(self.alphabet[char])
            aug_label.append(self.alphabet[''])
        return aug_label

    #def predict_best_path(self, output_timeseries):
def log_of_sum(a, b):
    """
        ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a))), unless one of them is zero
    """
    if (a != 0 and b != 0):
        return a + np.log(1 + np.exp(b - a))
    elif (a != 0):
        return a
    elif (b != 0):
        return b
    else:
        return 0

def prod_of_logs(a, b):
    """
        ln(ab) = ln(a) + ln(b), unless one of them is zero
    """
    if (a != 0 and b != 0):
        return a + b
    else:
        return 0


def test_eval_forward():
    # Note: run tests without rescale_alpha
    alphabet1 = {'c': 0, 'a' : 1, 't' : 2, 'd' : 3, 'o':  4, 'g': 5, '': 6}
    alphabet2 = {'h': 0, 'e' : 1, 'l' : 2, 'o' : 3, '' : 4}

    dec1 = CTCDecoder(alphabet1)
    dec2 = CTCDecoder(alphabet2)

    # Not valid distributions, but easy to compute
    output_timeseries_1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    # A bit more realistic
    output_timeseries_2 = np.array([[0.3, 0.2, 0.1, 0.3, 0.1], [0.1, 0.5, 0.1, 0.2, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2],
                                    [0.6, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.3, 0.2], [0.1, 0.1, 0.1, 0.3, 0.1],
                                    [0.1, 0.1, 0.1, 0.3, 0.1]])

    print np.isclose(dec1.eval_forward_prob(output_timeseries_1, "cat"), 0.0007, 1e-9)
    print np.isclose(dec1.eval_forward_prob(output_timeseries_1, "dog"), 0.0007, 1e-9)
    print np.isclose(dec2.eval_forward_prob(output_timeseries_2, "hello"), 0.0001344, 1e-9)
