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
        Assigns an integer value to all string entries. In particular:
         - in 'history_segment', the entries are replaced by an integer from 0 to 7(?)
         - in 'zip_code', Rural = 0, Suburban = 1 and Urban = 2
         - in 'channel', Multichannel = 0, Phone = 1, Web = 2
         - in 'segment', No E-Mail = 0, Mens E-Mail = 1, Womens E-Mail = 2
        """
        self.data["history_segment"] = self.data["history_segment"].factorize(sort=True)[0]
        self.data["zip_code"] = self.data["zip_code"].factorize(sort=True)[0]
        self.data["channel"] = self.data["channel"].factorize(sort=True)[0]
        self.data["segment"] = self.data["segment"].replace({"No E-Mail" : 0, "Mens E-Mail" : 1, "Womens E-Mail" : 2})

    def getCampaignData(self, campaign=1):
        """
        Returns data that is compatible with CausalML meta-learners
        :param campaign: 1 if we want to analyse the mens campaign, 2 for the womens campaign
        :return: a tuple of 3 arrays: the outcome (visit), the features (from recency to channel) and the treatment indicator(segment)
        """
        if campaign != 1 and campaign != 2:
            raise ValueError
        current_dataset = self.data[self.data["segment"] != 3 - campaign]
        outcome = current_dataset["visit"].to_numpy()
        features = current_dataset.iloc[:, :8].to_numpy()
        treatment = current_dataset["segment"].to_numpy()
        return outcome, features, treatment
