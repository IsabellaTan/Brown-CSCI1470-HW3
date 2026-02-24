import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        data = np.asarray(data)

        # Identify unique class labels
        self.classes_ = np.unique(data)

        # Create identity matrix for one-hot encoding
        identity = np.eye(len(self.classes_))

        # Map each label to a corresponding one-hot vector
        self.class_to_vec = {
            label: identity[i]
            for i, label in enumerate(self.classes_)
        }

        # Create reverse mapping: index -> label
        self.vec_to_class = {
            i: label
            for i, label in enumerate(self.classes_)
        }
        # return NotImplementedError

    def forward(self, data):
        data = np.asarray(data)

        # Automatically fit if not already fitted
        if not hasattr(self, "class_to_vec"):
            self.fit(data)

        # Convert each label to its one-hot vector
        one_hot = np.array([self.class_to_vec[label] for label in data])

        return one_hot
        # return NotImplementedError

    def inverse(self, data):
        data = np.asarray(data)

        # Get index of maximum value (class index)
        indices = np.argmax(data, axis=1)

        # Map indices back to original labels
        labels = np.array([self.vec_to_class[i] for i in indices])

        return labels
        # return NotImplementedError
