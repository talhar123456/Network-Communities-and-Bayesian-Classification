from collections import defaultdict
from typing import Dict, List, Set, Tuple


class ClassificationGroup:
    """ Stores data for a classification group. """
    def __init__(self, label: str):
        # the label of the classification group, e.g. healthy or disease
        self.label = label                              # type: str
        # how many valid samples were associated with the group in the training data set
        self.sample_count = 0                           # type: int
        # key: (feature, variant), value: how often the variant was observed for the feature in the training data set
        self.feature_counts = defaultdict(int)          # type: Dict[Tuple[int, str], int]
        # key: (feature, variant): value: observed frequency of the variant for the feature in the training data set
        self.likelihoods = defaultdict(float)           # type: Dict[Tuple[int, str], float]
        # prior and its log-version
        self.prior = 0.0                                # type: float
        self.log_prior = 0.0                            # type: float


class EvaluationResult:
    """ This class contains the results of evaluating a test data set with the NaiveBayesClassifier. """
    def __init__(self):
        # number of valid samples in the test data set
        self.valid_samples = 0                  # type: int
        # predicated classification of each valid sample in the test data set as one of the two groups
        # key: the ID of the sample, value: label of group_1 or group_2
        self.ground_truth = {}                  # type: Dict[str, str]
        self.prediction = {}                    # type: Dict[str, str]
        # frequency of each classification group in the test data set (only considering valid samples)
        # key: classification group label, value: frequency in the test data set
        self.frequency_ground_truth = {}        # type: Dict[str, float]
        self.frequency_predicted = {}           # type: Dict[str, float]
        # accuracy of the prediction compared to the ground truth contained in the test data set,
        # i.e. the ratio of correctly classified samples of all valid test samples
        self.accuracy = 0.0                     # type: float


class NaiveBayesClassifier:
    """
    This class builds a binary naive Bayes classifier from training data.
    """
    def __init__(self, label_1: str, label_2: str, n_features: int, variants: Set[str], training_data_path: str):
        """
        Reads training data from a file and builds the classification model. Your implementation should (at least) be
        able to:

        1) Ignore samples whose label does not match one of the input labels.
        2) Ignore samples with an incorrect number of features.
        3) Ignore samples where one of variants given for a feature is not in the set of allowed variants.
        4) Dynamically work with the labels and variant set provided as input, meaning no hard-coding of
           'healthy' and 'disease', or 'A', 'C', 'G', 'T'.

        :param label_1: label of the first classification group, e.g. healthy
        :param label_2: label of the second classification group, e.g. disease
        :param n_features: number of features
        :param variants: set of allowed variants, e.g. {'A', 'C', 'G', 'T'}
        :param training_data_path: path to a tab-separated file with
                                   column 1: sample ID, column 2: ground truth, columns 3+: features

        :raises: ValueError (with a custom message) if label_1 or label_2 are empty, or if they are the same
        :raises: ValueError (with a custom message) if a classification group has less than 3 samples
        """
    def __init__(self, label_1: str, label_2: str, n_features: int, variants: Set[str], training_data_path: str):
        self.n_features = n_features
        self.variants = variants
        self.group_1 = ClassificationGroup(label_1)
        self.group_2 = ClassificationGroup(label_2)
        self.training_data_size = 0
        self.log_ratios = {}

        with open(training_data_path, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) != n_features + 2:
                    continue
                label = data[1]
                if label == label_1:
                    group = self.group_1
                elif label == label_2:
                    group = self.group_2
                else:
                    continue
                group.sample_count += 1
                for i in range(2, len(data)):
                    feature = i - 2
                    variant = data[i]
                    if variant not in variants:
                        continue
                    group.feature_counts[(feature, variant)] += 1

        if self.group_1.sample_count < 3 or self.group_2.sample_count < 3:
            raise ValueError("Each classification group must have at least 3 samples.")

        self.training_data_size = self.group_1.sample_count + self.group_2.sample_count

        for feature in range(n_features):
            for variant in variants:
                count_1 = self.group_1.feature_counts[(feature, variant)]
                count_2 = self.group_2.feature_counts[(feature, variant)]
                ratio = (count_2 + 1) / (count_1 + 1)  # Laplace smoothing
                log_ratio = ratio  # log(count_2+1) - log(count_1+1)
                self.log_ratios[(feature, variant)] = log_ratio

        self.group_1.prior = self.group_1.sample_count / self.training_data_size
        self.group_2.prior = self.group_2.sample_count / self.training_data_size
        self.group_1.log_prior = self.group_1.sample_count / self.training_data_size
        self.group_2.log_prior = self.group_2.sample_count / self.training_data_size


    def get_group(self, label: str) -> ClassificationGroup:
        """
        :returns: the classification group associated with the label
        :raise: KeyError (with a custom message) if the label does not belong to one of the two groups
        """
        if label == self.group_1.label:
            return self.group_1
        elif label == self.group_2.label:
            return self.group_2
        else:
            raise KeyError("Label not found in the classification groups.")


    def highest_absolute_log_likelihood_ratios(self, n: int) -> List[Tuple[float, Tuple[int, str]]]:
        """
        Determine the (up to) n (feature, variant) combinations with the highest absolute log-likelihood ratios.

        Example:
        Feature: 5, variant: A, log ratio: -1.5
        Feature: 87, variant: G, log ratio: 1.4
        Feature: 5, variant: T, log ratio: -0.8

        :param n: top number of (feature, variant) combinations
        :return: list of (feature, variant) combinations with the highest absolute log-likelihood ratios (in
                 descending order by their absolute ratio) as (log-likelihood ratio, (feature index, variant)) tuples,
                 e.g. [(-1.5, (5, 'A')), (1.4, (87, 'G')), (-0.8, (5, 'T')),...]
        :raise: ValueError (with a custom message) if n is negative
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        sorted_ratios = sorted(self.log_ratios.items(), key=lambda x: abs(x[1]), reverse=True)
        return [(ratio, feature) for feature, ratio in sorted_ratios[:n]]

    def classify(self, sample: List[str]) -> str:
        """
        Classify a sample based on its features as one of the two groups.

        :param sample: list of variants, e.g. ['A', 'G',... , 'C'] (one variant for each feature)
        :return: group classification label
        :raises: ValueError (with a custom message) if the number of features in the sample does not match the expected
                 number of features
        :raises: ValueError (with a custom message) if an element of the sample does not match the allowed variants
        """
        log_likelihood_1 = self.group_1.log_prior
        log_likelihood_2 = self.group_2.log_prior
        for feature, variant in enumerate(sample):
            if variant not in self.variants:
                raise ValueError("Invalid variant in sample.")
            log_ratio = self.log_ratios.get((feature, variant), 0.0)
            log_likelihood_1 += log_ratio
            log_likelihood_2 -= log_ratio
        if log_likelihood_1 > log_likelihood_2:
            return self.group_1.label
        else:
            return self.group_2.label

    def evaluate_accuracy(self, test_data_path: str) -> EvaluationResult:
        """
        Classify each sample in the test data based on its features and compare the classification to the ground truth
        contained in the test file, in order to determine the overall accuracy. See the specifications in the
        initialization function for determining which samples in the test data are valid.

        :return: Evaluation-object with the results of evaluating the test data
        :param test_data_path: path to a tab-separated file with
                               column 1: sample ID, column 2: ground truth, columns 3+: features
        :raises: ValueError (with a custom message) if the test data is empty
        """
        result = EvaluationResult()
        with open(test_data_path, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) != self.n_features + 2:
                    continue
                sample_id, ground_truth = data[0], data[1]
                if ground_truth not in [self.group_1.label, self.group_2.label]:
                    continue
                sample = data[2:]
                prediction = self.classify(sample)
                result.valid_samples += 1
                result.ground_truth[sample_id] = ground_truth
                result.prediction[sample_id] = prediction
                result.frequency_ground_truth[ground_truth] = result.frequency_ground_truth.get(ground_truth, 0) + 1
                result.frequency_predicted[prediction] = result.frequency_predicted.get(prediction, 0) + 1
                if ground_truth == prediction:
                    result.accuracy += 1
        if result.valid_samples == 0:
            raise ValueError("Test data is empty.")
        result.accuracy /= result.valid_samples
        return result


def exercise_2():
    def evaluate(classifier, training_set, test_set):
        print(f"Results for {training_set} and {test_set}:")
        # Instantiate the NaiveBayesClassifier
        classifier = NaiveBayesClassifier(label_1="healthy", label_2="disease", n_features=100,
                                           variants={'A', 'C', 'G', 'T'}, training_data_path=training_set)

        # Get the classification groups
        group_1 = classifier.get_group("healthy")
        group_2 = classifier.get_group("disease")

        # Print the prior probabilities
        print("Prior probability of", group_1.label, ":", group_1.prior)
        print("Prior probability of", group_2.label, ":", group_2.prior)

        # Get and print the top 5 highest absolute log-likelihood ratios
        top_5_ratios = classifier.highest_absolute_log_likelihood_ratios(5)
        print("Top 5 highest absolute log-likelihood ratios:")
        for ratio, (feature, variant) in top_5_ratios:
            print("Feature:", feature, ", Variant:", variant, ", Log-likelihood ratio:", ratio)

        # Classify a sample
        sample = ['A', 'C', 'G', 'T', 'G', 'G', 'A', 'C', 'C', 'T', 'T', 'A', 'G', 'A', 'T', 'C', 'G', 'A', 'C', 'T']
        classification = classifier.classify(sample)
        print("Sample classified as:", classification)

        # Evaluate accuracy using test data
        evaluation_result = classifier.evaluate_accuracy(test_data_path=test_set)
        print("Accuracy:", evaluation_result.accuracy)
        print("Frequency of ground truth labels:", evaluation_result.frequency_ground_truth)
        print("Frequency of predicted labels:", evaluation_result.frequency_predicted)
        print()

    evaluate(NaiveBayesClassifier, "training_set_1.tsv", "test_set_1.tsv")
    evaluate(NaiveBayesClassifier, "training_set_2.tsv", "test_set_2.tsv")


if __name__ == "__main__":
    exercise_2()
