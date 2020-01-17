"""bangdataresult.py: Bang data results module
********************************************
Author: Irena & Junior
This module encapsulates the results class for the Scaled Humanity Bang Data Pipeline
"""

import pandas as pd
pd.set_option('display.max_colwidth', -1)

### BANGDATARESULT CLASS ###
class BangDataResult():
    """ This class serves as the viewer for the data pipeline.
    It holds one batch's results as fed to it by BangDataPipeline.
    We completely separate this obj from analysis because I foresee
    people messing with this class a lot and I don't want analysis 
    to be only carried out once, and correctly. """

    def __init__(self, batch_id: str, json=None, df=None, t_df=None, u_df=None):
        ## PUBLIC FIELDS ##
        self.batch = batch_id
        self.users = u_df.index
        self.teams = t_df.index
        self.numRounds = json['numRounds']
        self.refPair1 = sorted(json['expRounds']) # always [ref, reconvene]
        self.refPair2 = sorted(json['worstRounds']) if 'worstRounds' in json and len(json['worstRounds']) != 0 \
            else [self.refPair1[0], self.numRounds - (self.refPair1[1] == self.numRounds)]
            # will either be [ref2, reconvene2] or [ref, control]
        self.labels = ["reconvene", "control"] if self.refPair1[0] == self.refPair2[0] else ["highest", "lowest"]
        
        ## PRIVATE FIELDS ##
        self._json = json
        self._raw_df = df
        self._team_df = t_df
        self._user_df = u_df
        self._analyses = {}

    ## SETTERS ##
    def set(self, key: str, team, ind=None):
        """ ideally not a public method """
        self._analyses[key + "_TEAM"] = team
        if (ind is not None):
            self._analyses[key + "_IND"] = ind

    ## UTILS ##
    def combine(self, table1, table2):
        """ combines two tables OF SAME SIZE by zipping cells """
        # make sure the same size
        if len(table1.index) != len(table2.index): return
        if len(table1.columns) != len(table2.columns): return

        combine = pd.DataFrame(index=table1.index, columns=table1.columns)
        for i in range(len(table1.index)):
            combine.iloc[i] = list(zip(table1.iloc[i], table2.iloc[i])) 

        return combine

    def parse_chat(self, team_chat):
        """ public: given a chat json object, return a neat formatted df 
        of the chat transcriptions, minus helperbot """
        chat = pd.DataFrame(team_chat[0])  # [0] is bc weird nested list thing
        chat = chat[['user', 'nickname', 'message', 'time']]
        chat = chat.sort_values(by="time")
        chat = chat[chat['user'] != '100000000000000000000001']  # remove bot
        return chat

    ## GETTERS ##
    def json(self):
        return self._json

    def raw_df(self):
        return self._raw_df

    def team_df(self):
        return self._team_df['user']

    def raw_chats(self):
        return self._team_df['chat']

    def user_df(self):
        return self._user_df

    def viability(self, ind=False, block=False):
        """ returns the viability table, default at team-level, with added diff columns """
        viability =  self._analyses["VIABILITY_IND"].copy() if ind \
            else self._analyses["VIABILITY_TEAM"].copy(deep=True)
        if ind: return viability

        # enforce block (kick out people who missed a survey at any point)
        if block:
            # set to none to kick out people who missed a survey
            viability['user'] = viability['user'].apply(lambda u: u if u.count() == self.numRounds \
                else [None] * self.numRounds, axis=0)
            viability['mean_viability'] = viability['user'].mean(axis=1)

        # get diffs here, team only
        viability[self.labels[0]] = - \
            (viability['mean_viability'] - viability.iloc[self.refPair1[1]-1]['mean_viability'].item())
        viability[self.labels[1]] = - \
            (viability['mean_viability'] - viability.iloc[self.refPair2[1]-1]['mean_viability'].item())

        return viability
    
    def fracture(self, ind=False):
        """ returns the fracture table, default at team-level """
        return self._analyses["FRACTURE_IND"] if ind else self._analyses["FRACTURE_TEAM"]
    
    def fracture_why(self, ind=False):
        """ returns the fracture table, default at team-level """
        return self._analyses["FRACTURE_WHY_IND"] if ind else self._analyses["FRACTURE_WHY_TEAM"]
    
    def chat(self, ind=False):
        """ returns the chat composition table, default at team-level """
        return self._analyses["CHAT_IND"] if ind else self._analyses["CHAT_TEAM"]
    
    def manipulation(self):
        return self._analyses["MANIPULATION_TEAM"]

    def display(self):
        return list(self._analyses.values())
    
    def __str__(self):
        return str(list(self._analyses.values()))

