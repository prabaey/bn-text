from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling.Sampling import BayesianModelSampling

class DiseaseModelGT():

    """
    Create the ground truth DAG with expert-defined conditional probability tables
    """

    def __init__(self):
    
        # DAG
        self.model = BayesianNetwork([("season", "pneu"), ("season", "inf"),
                                      ("pneu", "dysp"), ("pneu", "cough"), ("pneu", "fever"), ("pneu", "pain"),
                                      ("inf", "cough"), ("inf", "fever"), ("inf", "nasal"), ("inf", "pain")])
        
        # background 
        cpd_season = TabularCPD(variable="season", variable_card = 2, values = [[0.4], [0.6]], state_names={"season": ["cold", "warm"]})
        
        # diagnoses
        cpd_pneu =  TabularCPD(variable="pneu", variable_card = 2, values = [[0.015, 0.005],
                                                                            [0.985, 0.995]], 
                                evidence = ["season"], evidence_card = [2], state_names={"season": ["cold", "warm"], "pneu": ["yes", "no"]})
        cpd_inf =   TabularCPD(variable="inf", variable_card = 2, values = [[0.5, 0.05], 
                                                                            [0.5, 0.95]], 
                                evidence = ["season"], evidence_card = [2], state_names = {"season": ["cold", "warm"], "inf":["yes", "no"]})
        
        # partially observed symptoms
        cpd_dysp =  TabularCPD(variable="dysp", variable_card = 2, values = [[0.3, 0.15],
                                                                             [0.7, 0.85]], 
                                evidence = ["pneu"], evidence_card = [2], state_names={"dysp": ["yes", "no"], "pneu": ["yes", "no"]})
        cpd_cough = TabularCPD(variable="cough", variable_card = 2, values = [[0.9, 0.9, 0.8, 0.05], #(pneu=yes,inf=yes), (pneu=yes,inf=no), (pneu=no, inf=yes), (pneu=no, inf=no)
                                                                              [0.1, 0.1, 0.2, 0.95]], 
                                evidence = ["pneu", "inf"], evidence_card = [2, 2], state_names={"cough": ["yes", "no"], "pneu": ["yes", "no"], "inf": ["yes", "no"]})
        cpd_nasal = TabularCPD(variable="nasal", variable_card = 2, values = [[0.7, 0.2],
                                                                              [0.3, 0.8]], 
                                evidence = ["inf"], evidence_card = [2], state_names={"nasal": ["yes", "no"], "inf": ["yes", "no"]})

        # unobserved symptoms
        cpd_fever = TabularCPD(variable="fever", variable_card = 3, values = [[0.05, 0.10, 0.85, 0.80], #(pneu=yes,inf=yes), (pneu=yes,inf=no), (pneu=no, inf=yes), (pneu=no, inf=no)
                                                                              [0.15, 0.10, 0.14, 0.15],
                                                                              [0.80, 0.80, 0.01, 0.05]], 
                                evidence = ["pneu", "inf"], evidence_card = [2, 2], state_names={"fever": ["none", "low", "high"], "pneu": ["yes", "no"], "inf": ["yes", "no"]})
        cpd_pain =  TabularCPD(variable="pain", variable_card = 2, values = [[0.3, 0.3, 0.1, 0.05], #(pneu=yes,inf=yes), (pneu=yes,inf=no), (pneu=no, inf=yes), (pneu=no, inf=no)
                                                                             [0.7, 0.7, 0.9, 0.95]], 
                                evidence = ["pneu", "inf"], evidence_card = [2, 2], state_names={"pain": ["yes", "no"], "pneu": ["yes", "no"], "inf": ["yes", "no"]})
        
        self.model.add_cpds(cpd_season, cpd_pneu, cpd_inf, cpd_dysp, cpd_cough, cpd_nasal, cpd_fever, cpd_pain)

        self.class_map = {"season": {"warm": 0, "cold": 1}, 
                          "pneu": {"no": 0, "yes": 1}, "inf": {"no": 0, "yes": 1}, 
                          "dysp": {"no": 0, "yes": 1}, "cough": {"no": 0, "yes": 1}, "nasal": {"no": 0, "yes": 1},
                          "fever": {"none": 0, "low": 1, "high": 2}, "pain": {"no": 0, "yes": 1}}

    
    def sample(self, N, seed=666):
        """
        get N samples from the Bayesian network
        """

        samp_obj = BayesianModelSampling(self.model)
        samples = samp_obj.forward_sample(size=N, seed=seed)

        return samples
    
from torch.utils.data import Dataset
import torch
import numpy as np 
import pandas as pd

class TabDataset(Dataset):
    """
    Tabular dataset for use in Pytorch models
    """
    def __init__(self, dataframe, class_map, emb_type, device):
        """
        dataframe: dataset containing samples made up of values for the background, disease and symptom variables, as well as text embeddings
        class_map: dictionary mapping class names for each variable to index {"var_name": {"class_name": int}}
        emb_type: name of embedding to use, will be "BioLORD emb" in our models
        device: device to load tensors to
        """
        
        self.dataframe = dataframe
        self.class_map = class_map
        self.emb_type = emb_type
        self.device = device

        self.preprocessed_data = self.preprocess()

    def __len__(self):
        """
        returns number of samples in dataset
        """

        return len(self.preprocessed_data["index"])

    def preprocess(self):
        """
        turn dataframe of length n into a dictionary of tensors 
        use class_map to map variable classes (e.g. "no"/"yes") to float (e.g. 0.0/1.0). use np.nan for unobserved symptom values
        dict contains {"season": tensor(), "pneu": tensor(), "inf": tensor(), "dysp": tensor(), "cough":tensor(), "nasal":tensor(), "text":tensor()}
        each tensor is one-dimensional with length n, except for "text", which has dimension (n, 768)
        """
        
        preprocessed_data = {}

        for var in self.class_map: 
            values = self.dataframe[var].apply(lambda x: self.class_map[var][x] if not pd.isna(x) else np.nan).values
            preprocessed_data[var] = torch.tensor(values, dtype=torch.float, device=self.device)

        preprocessed_data["text"] = torch.tensor(self.dataframe[self.emb_type].tolist(), dtype=torch.float, device=self.device)
        preprocessed_data["index"] = torch.tensor(self.dataframe.index.tolist(), device=self.device)

        return preprocessed_data
    
    def len_sympt_obs(self): 
        """
        get the number of samples in the dataset where the symptoms are observed (not nan)
        """
        
        df_sympt_obs = self.dataframe.dropna(axis=0, how="any") # drop all records where symptoms not observed
        return len(df_sympt_obs)

    def __getitem__(self, index):
        """
        return sample {"season": tensor(), "pneu": tensor(), "inf": tensor(), "dysp": tensor(), "cough":tensor(), "nasal":tensor(), "text": tensor()}
        """

        # sample = self.dataframe.iloc[index]
        # dp = {}
        # for var in self.class_map: 
        #     if pd.isna(sample[var]):
        #         val = torch.tensor(np.nan, dtype=torch.float, device=self.device)
        #     else: 
        #         val = torch.tensor(self.class_map[var][sample[var]], dtype=torch.float, device=self.device)
        #     dp[var] = val


        # dp["text"] = torch.tensor(sample[self.emb_type], device=self.device) # convert embedding to tensor
        # dp["index"] = torch.tensor(index, device=self.device) # index to connect back to original dataframe

        dp = {var: self.preprocessed_data[var][index] for var in self.preprocessed_data.keys()}
        
        return dp