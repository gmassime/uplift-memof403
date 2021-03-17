import pandas as pd
from abc import ABC, abstractmethod


class DatasetCSV(ABC):

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.cleanDataset()

    @abstractmethod
    def cleanDataset(self):
        pass

    @abstractmethod
    def getTrainTest(self):
        pass

    @abstractmethod
    def getFeatureNames(self):
        pass


class HillstromDataset(DatasetCSV):

    def __init__(self,
                 file_path="http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"):
        """
        Initializes a dataset using data from the Hillstrom MineThatData E-Mail Analytics And Data Mining Challenge
        :param file_path: path of the .csv file with the dataset, by default it uses the URL of the web page containing the dataset
        """
        super().__init__(file_path)

    def cleanDataset(self):
        """
        Prepares the dataset for use. In particular:
         - 'history_segment', 'zip_code' and 'channel' are transformed with one-hot encoding
         - in 'segment', No E-Mail = 0, Mens E-Mail = 1, Womens E-Mail = 2
         - 'spend' and 'conversion' are dropped, we will only target 'visit'
        """
        for column in ["history_segment", "zip_code", "channel"]:
            one_hot = pd.get_dummies(self.data[column])
            self.data = self.data.drop(column, axis=1).join(one_hot)
        self.data["segment"] = self.data["segment"].replace({"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2})
        self.data = self.data.drop(["spend", "conversion"], axis=1)

    def getTrainTest(self, campaign=1, frac=0.5, seed=0):
        """
        Returns data that is compatible with CausalML meta-learners
        :param campaign: 1 if we want to analyse the mens campaign, 2 for the womens campaign
        :param frac: fraction of population in training set
        :param seed: seed for shuffling the data
        :return: 6 arrays, 3 for the train set and 3 for the test set: the outcome (visit), the features and the treatment indicator (segment)
        """
        if campaign != 1 and campaign != 2:
            raise ValueError('Invalid campaign')
        if frac < 0 or frac > 1:
            raise ValueError('Invalid fraction')

        train = self.data[self.data["segment"] != 3 - campaign].sample(frac=frac, random_state=seed).replace({campaign : 1})
        test = self.data[self.data["segment"] != 3 - campaign].drop(train.index).sample(frac=1.0).replace({campaign : 1})   # to ensure that data is shuffled

        outcome_train = train["visit"].to_numpy()
        features_train = train.drop(["visit", "segment"], axis=1).to_numpy()
        treatment_train = train["segment"].to_numpy()

        outcome_test = test["visit"].to_numpy()
        features_test = test.drop(["visit", "segment"], axis=1).to_numpy()
        treatment_test = test["segment"].to_numpy()

        return outcome_train, features_train, treatment_train, outcome_test, features_test, treatment_test

    def getFeatureNames(self):
        """
        Returns the names of the features (used to visualize the uplift tree in CausalML)
        :return: array of string
        """
        return list(self.data.drop(["visit", "segment"], axis=1))


class CriteoDataset(DatasetCSV):

    def __init__(self, file_path="http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"):
        """
        Initializes a dataset using the Criteo Uplift Prediction data
        :param file_path: path of the .csv file with the dataset, by default it uses the URL of the web page containing the dataset
        """
        super().__init__(file_path)

    def cleanDataset(self):
        """
        Prepares the dataset for use. In particular:
         - 'spend' and 'conversion' are dropped, we will only target 'visit'
        """
        self.data = self.data.drop(["conversion", "exposure"], axis=1)

    def getTrainTest(self, frac=0.5, seed=0):
        """
        Returns data that is compatible with CausalML meta-learners
        :param frac: fraction of population in training set
        :param seed: seed for shuffling the data
        :return: 6 arrays, 3 for the train set and 3 for the test set: the outcome (visit), the features and the treatment indicator
        """

        train = self.data.sample(frac=frac, random_state=seed)
        test = self.data.drop(train.index).sample(frac=1.0)

        outcome_train = train["visit"].to_numpy()
        features_train = train.drop(["visit", "treatment"], axis=1).to_numpy()
        treatment_train = train["treatment"].to_numpy()

        outcome_test = test["visit"].to_numpy()
        features_test = test.drop(["visit", "treatment"], axis=1).to_numpy()
        treatment_test = test["treatment"].to_numpy()

        return outcome_train, features_train, treatment_train, outcome_test, features_test, treatment_test

    def getFeatureNames(self):
        """
        Returns the names of the features (used to visualize the uplift tree in CausalML)
        :return: array of string
        """
        return list(self.data.drop(["visit", "treatment"], axis=1))
