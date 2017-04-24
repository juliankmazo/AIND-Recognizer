import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        x = []
        y = []
        best_n = 2

        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                bic = 0
                count = 0
                min_splits = min(len(self.lengths), 3)
                split_method = KFold(n_splits=min_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        x_train = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l_train = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            n_iter=1000,
                                            random_state=self.random_state)\
                            .fit(x_train, l_train)

                        x_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        bic += -2 * model.score(x_test, l_test) + 2 * n * len(self.sequences[0])\
                            * np.log(np.array(self.lengths)[cv_test_idx].sum())
                        count += 1
                    except:
                        pass

                if count > 0 and bic != 0:
                    y.append(bic / count)
                    x.append(n)

        if len(y) > 0:
            best_n = x[y.index(min(y))]

        return GaussianHMM(n_components=best_n,
                           n_iter=1000,
                           random_state=self.random_state)\
            .fit(self.X, self.lengths)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        x = []
        y = []
        best_n = 2

        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                count = 0
                dic = 0
                log_l = 0
                min_splits = min(len(self.lengths), 3)
                split_method = KFold(n_splits=min_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        x_train = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l_train = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            n_iter=1000,
                                            random_state=self.random_state)\
                            .fit(x_train, l_train)

                        x_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        log_l = model.score(x_test, l_test)

                        count_d_l = 0
                        log_d_l = 0
                        correction = 0

                        for w in self.words.keys():
                            if w != self.this_word:
                                log_d_l += model.score(self.hwords[w][0], self.hwords[w][1])
                                count_d_l += 1
                                correction += 2 * n * len(self.sequences[0]) / 2 * \
                                    np.log(sum(self.hwords[w][1]) / np.array(self.lengths)[cv_test_idx].sum())

                        dic += log_l - log_d_l / (count_d_l - 1) + correction
                        count += 1
                    except:
                        pass

                if count > 0 and dic != 0:
                    y.append(dic / count)
                    x.append(n)

        if len(y) > 0:
            best_n = x[y.index(max(y))]

        return GaussianHMM(n_components=best_n,
                           n_iter=1000,
                           random_state=self.random_state).fit(self.X, self.lengths)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        x = []
        y = []
        best_n = 2

        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                count = 0
                log_l = 0
                min_splits = min(len(self.lengths), 3)
                split_method = KFold(n_splits=min_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        x_train = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l_train = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            n_iter=1000,
                                            random_state=self.random_state)\
                            .fit(x_train, l_train)

                        x_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        log_l += model.score(x_test, l_test)
                        count += 1
                    except:
                        pass

                if count > 0 and log_l != 0:
                    y.append(log_l / count)
                    x.append(n)

        if len(y) > 0:
            best_n = x[y.index(max(y))]

        return GaussianHMM(n_components=best_n, n_iter=1000, random_state=self.random_state).fit(self.X, self.lengths)
