import pandas as pd

class HillstromDataset:

    def __init__(self, file_path="http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"):
        """
        Initializes a dataset using Hillstrom data
        :param file_path: path of the .csv file with the dataset, by default it uses the URL of the web page containing the dataset
        """
        self.data = pd.read_csv(file_path)
        self.cleanDataset()

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
        self.data["segment"] = self.data["segment"].replace({"No E-Mail" : 0, "Mens E-Mail" : 1, "Womens E-Mail" : 2})
        self.data = self.data.drop(["spend", "conversion"], axis=1)

    def getCampaignData(self, campaign=1):
        """
        Returns data that is compatible with CausalML meta-learners
        :param campaign: 1 if we want to analyse the mens campaign, 2 for the womens campaign
        :return: a tuple of 3 arrays: the outcome (visit), the features and the treatment indicator (segment)
        """
        if campaign != 1 and campaign != 2:
            raise ValueError
        current_dataset = self.data[self.data["segment"] != 3 - campaign]
        outcome = current_dataset["visit"].to_numpy()
        features = current_dataset.drop(["visit", "segment"], axis=1).to_numpy()
        treatment = current_dataset["segment"].to_numpy()
        return outcome, features, treatment
