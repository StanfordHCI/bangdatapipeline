"""bangdata.py: Bang data processing module
********************************************
Author: Irena & Junior & Tonya
This module encapsulates the analysis engine for the Scaled Humanity Bang Data Pipeline
"""

### IMPORTS ###

import requests
import json
from json import loads
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)

__version__ = "0.1.0"


### BANGDATAPIPELINE CLASS ###
class BangDataPipeline():
    """ This class serves as the controller and modeller for the data pipeline.
    It fetches data and returns a BangDataResults object that contains the
    processed, analyzed data. """

    def __init__(self, token, survey_settings={"LENGTH": 0}, verbose=True):
        self._verbose = verbose
        # default survey_settings is blank
        self._survey_settings = survey_settings
        self._token = token

        if verbose:
            print("Initialized BangDataPipeline. Set up to analyze surveys with this configuration:\n")
            print(survey_settings)

    ## MODIFY BANG DATA PIPELINE SETTINGS ##
    def set_verbose(self, val):
        """ sets verbose flag to new value """
        self._verbose = bool(val)

        if self._verbose:
            print(f"verbose was set to {val}.")

    def __set_survey_setting(self, key, val):
        """ not a public method """
        old_val = self._survey_settings.get(key, "UNSET")
        self._survey_settings[key] = val

        if self._verbose:
            print(
                f"survey setting {key} was updated. Old value was {old_val}", end=" ")
            print(f". New value is set to {val}", end=" ")

    def set_survey_len(self, len: int):
        """ updates survey settings dictionary with new
        survey len values """
        # quick error checking
        if len >= 0:
            self.__set_survey_setting("LENGTH", len)

    def set_viability_index(self, start: int, end: int):
        """ updates survey settings dictionary with new
        viability question index values """
        # quick error checking
        if end > start and start >= 0 and end < self._survey_settings["LENGTH"]:
            self.__set_survey_setting("VIABILITY_START", start)
            self.__set_survey_setting("VIABILITY_END", end)

    def set_fracture_index(self, index: int):
        """ updates survey settings dictionary with new
        fracture question index values """
        # quick error checking
        if index >= 0 and index < self._survey_settings["LENGTH"]:
            self.__set_survey_setting("FRACTURE_INDEX", index)

    def set_fracture_why_index(self, index: int):
        """ updates survey settings dictionary with new
        fracture question index values """
        # quick error checking
        if index >= 0 and index < self._survey_settings["LENGTH"]:
            self.__set_survey_setting("FRACTURE_WHY_INDEX", index)

    ## REQUEST FUNCTIONS ##

    def __valid(self, json):
        """ given a batch json file, returns whether or not it was a completed valid batch """
        try: 
            return json['status'] == "completed" and json['currentRound'] == json['numRounds']
        except: 
            return False

    def fetch(self):
        """ public: fetch all recent batch metadata and display in table """
        # make request to server
        url = "https://bang.stanford.edu:3001/api/admin/batches/"
        querystring = {"search.sessiontype": "1522435540042001BxTD"}
        payload = {"key": "value"}
        headers = {
            'Accept': "application/json, text/plain, */*",
            'admintoken': self._token,
            'Origin': "https://bang.stanford.edu",
            'Referer': "https://bang.stanford.edu/batches",
        }

        batchlist_request = requests.request(
            "GET", url, json=payload, headers=headers, params=querystring)
        batchlist_json = batchlist_request.json()
        batches_df = pd.DataFrame(
            columns=['batch_id', 'template', 'mask type', 'team format', 'team size', 'note'])
        i = 1
        for batch in batchlist_json['batchList']:
            if self.__valid(batch):
                batches_df.loc[i] = [batch['_id'], batch['templateName'],
                                     batch['maskType'], '', batch['teamSize'], '']
                if ('teamFormat' in batch):
                    batches_df.loc[i, 'team format'] = batch['teamFormat']
                if ('note' in batch):
                    batches_df.loc[i, 'note'] = batch['note']
                i += 1

        # output
        if self._verbose:
            print(f"Fetched all recent complete batches.")

        return batches_df

    def get_json(self, batch_id: str):
        """ public: for a batch_id, gets json data """
        # make request to server
        this_url = "https://bang-prod.deliveryweb.ru:3001/api/admin/batch-result/" + batch_id
        querystring = {"search.sessiontype": "1522435540042001BxTD"}
        payload = {"key": "value"}
        this_header = {
            'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
            'admintoken': self._token,
            'Origin': "https://bang-prod.deliveryweb.ru",
            'Referer': "https://bang-prod.deliveryweb.ru:3001/batches/" + batch_id,
        }
        get_request = requests.request(
            "GET", this_url, json=payload, headers=this_header, params=querystring)
        this_json = get_request.json()

        # output
        if self._verbose:
            print(f"Found batch {batch_id}. Returning raw json.")

        return this_json['batch']

    def get_jsons(self, batch_ids: [str]):
        """ public: for a list of batch_ids, gets json data """
        return [self.get_json(id) for id in batch_ids]

    ## ANALYSIS FUNCTIONS ##
    # SETUP ANALYSIS #
    def __batch_df(self, json):
        """ creates a table with one row per survey entry per user per team per round
        batch_df takes in a batch json and returns a dataframe representing that batch.
        df = round | team id | user id | survey results | chat """
        df = pd.DataFrame(columns=['round', 'team', 'user', 'survey', 'chat'])
        i = 1
        for round in json['rounds']:
            round_id = round['number']
            for team in round['teams']:
                team_id = team['_id']
                chat = team['chat']['messages']
                for user in team['users']:
                    user_id = user['user']
                    mid_survey = user['midSurvey']['questions'] if 'midSurvey' in user else None
                    df.loc[i] = [round_id, team_id, user_id,
                                mid_survey, chat]
                    i += 1
        return df

    def __team_df(self, df):
        """ creates a (round,team)-indexed user#-named table with userIDs
        This is mostly a helper for later functions
        t_df = team | user1 | ... | usern | chat """
        t_df = df[['user', 'round', 'team']]
        pd.options.mode.chained_assignment = None  # default='warn'
        t_df['round_team'] = list(zip(df['round'], df['team']))
        t_df = t_df[['user', 'round_team']]
        t_df['n'] = t_df.groupby('round_team').cumcount().add(1)
        t_df = t_df.set_index(['round_team', 'n'])
        t_df = t_df.unstack()

        t_df['chat'] = t_df.apply(
            lambda r: df[df['team'] == r.name[1]]['chat'].iloc[0], axis=1)
        return t_df

    def __user_df(self, df):
        """ creates user-indexed round-named table with raw mid-survey jsons
        This is mostly a helper for later functions
        u_df = user | round1survey object, unformatted | ... | roundmsurvey """
        u_df = df[['user', 'round', 'survey']
                  ].sort_values(by=['user', 'round'])
        u_df = u_df.pivot(index='round', columns='user', values='survey')
        u_df = u_df.transpose()
        return u_df

    def __get_survey(self, u_df, round, userid):
        """ helper function to return a survey object given a user and a round """
        return u_df.loc[userid, round]

    # VIABILITY ANALYSES #
    def __ind_viability(self, survey):
        """ given an uparsed survey object (one cell of u_df) return the 
        viability score by summing viability question answers """
        if (self._survey_settings["VIABILITY_START"] == None 
            or survey == None
            or survey != survey 
            or len(survey) < self._survey_settings["LENGTH"]):
            return(None)

        sum_viability = 0
        for i in range(self._survey_settings["VIABILITY_START"], self._survey_settings["VIABILITY_END"] + 1):
            sum_viability += int(survey[i]['result'])
        
        len_viability = self._survey_settings["VIABILITY_END"] - self._survey_settings["VIABILITY_START"]
        return sum_viability if sum_viability > len_viability else None #min is 1pt per question so cannot be 0

    def __analyze_viability_team(self, u_df, t_df):
        """ get team-indexed viability scores with average and diff """
        viability = t_df.apply(lambda team: team[:-1].apply(lambda userid: self.__ind_viability(
            self.__get_survey(u_df, team.name[0], userid)) if(userid == userid) else None), axis=1)
        # translation: across rows of t_df minus chat (so user ids), use the id and the round number (team.name[0]) to query ind_viability
      
        viability['mean_viability'] = viability.mean(axis=1)
        return viability

    def __analyze_viability_ind(self, u_df):
        """ get user-indexed round-named viability scores """
        viability = u_df.applymap(self.__ind_viability)
        return viability

    # FRACTURE ANALYSES #
    def __ind_fracture(self, survey):
        """given an unparsed survey json (one cell of u_df), 
        parse whether the person answered keep or not keep"""
        if(survey == None or survey != survey or len(survey) < self._survey_settings["LENGTH"]):
            return(None)

        fracture_res = "KEEP" if(int(
            survey[self._survey_settings["FRACTURE_INDEX"]]['result']) == 0) else "DO NOT KEEP"
        return(fracture_res)

    def __analyze_fracture_team(self, u_df, t_df):
        """ get team-indexed fracture results """
        fracture = t_df.apply(lambda team: team[:-1].apply(lambda userid: self.__ind_fracture(
            self.__get_survey(u_df, team.name[0], userid)) if(userid == userid) else None), axis=1)
        # translation: across rows of t_df minus chat (so user ids), use the id and the round number (t_df name) to query fracture
        return fracture

    def __analyze_fracture_ind(self, u_df):
        """ get user-indexed round-named fracture results """
        fracture = u_df.applymap(self.__ind_fracture)
        return fracture

    def __ind_fracture_why(self, survey):
        """given an unparsed survey json (one cell of u_df), 
        parse whether the person answered keep or not keep"""
        if(survey == None or survey != survey or len(survey) < self._survey_settings["LENGTH"]):
            return(None)

        fracture_res = survey[self._survey_settings["FRACTURE_WHY"]]['result']
        return(fracture_res)

    def __analyze_fracture_why_team(self, u_df, t_df):
        """ get team-indexed fracture results """
        fracture = t_df.apply(lambda team: team[:-1].apply(lambda userid: self.__ind_fracture_why(
            self.__get_survey(u_df, team.name[0], userid)) if(userid == userid) else None), axis=1)
        # translation: across rows of t_df minus chat (so user ids), use the id and the round number (t_df name) to query fracture
        return fracture

    def __analyze_fracture_why_ind(self, u_df):
        """ get user-indexed round-named fracture results """
        fracture = u_df.applymap(self.__ind_fracture_why)
        return fracture

    # MANIPULATION ANALYSIS FUNCTIONS #
    def __analyze_manipulation(self, raw):
        """ wrapper function for anlayzing manipulation. switches between 
        the single and team functions depending on teamFormat """
        if raw["teamFormat"] == "single": return self.__analyze_manipulation_single(raw)
        else: return self.__analyze_manipulation_multi(raw)

    def __analyze_manipulation_multi(self, raw):
        """ get manipulation check answer. for old multi team format """
        correct = raw['expRounds']

        # build up a manipulation df m_df from scratch, with columns user, guess1, guess2, and reasoning
        m_df = pd.DataFrame(columns=['user', 'mc1', 'mc2', 'why'])
        i = 1
        for user in raw['users']:
            user_id = user['user']['_id']
            if('survey' not in user or 'mainQuestion' not in user['survey']):
                m_df.loc[i] = [user_id, None, None, None]
            else:
                expRound1 = user['survey']['mainQuestion']['expRound1']
                expRound2 = user['survey']['mainQuestion']['expRound2']
                why = user['survey']['mainQuestion']['expRound3']
                m_df.loc[i] = [user_id, expRound1, expRound2, why]
            i += 1

        # fill in extra column with whether they were correct
        m_df["correct"] = m_df.apply(lambda row: sorted(
            [row['mc1'], row['mc2']]) == sorted(correct), axis=1)
        return m_df

    def __analyze_manipulation_single(self, raw):
        """ get manipulation check answer. for old single team format """

        # build up a manipulation df m_df from scratch, with columns user, guess1, guess2, and reasoning
        m_df = pd.DataFrame(columns=['user', 'guessedName', 'actualName', 'numOptions'])
        i = 1
        for user in raw['users']:
            user_id = user['user']['_id']
            if('survey' not in user or 'singleTeamQuestion' not in user['survey']):
                m_df.loc[i] = [user_id, None, None, None]
            else:
                guessed = user['survey']['singleTeamQuestion']['chosenPartnerName']
                actual = user['survey']['singleTeamQuestion']['actualPartnerName']
                num = user['survey']['singleTeamQuestion']['numOptions']
                m_df.loc[i] = [user_id, guessed, actual, num]
            i += 1

        # fill in extra column with whether they were correct
        if len(m_df.index) != 0:
            m_df["correct"] = m_df.apply(lambda row: row["guessedName"] == row["actualName"] 
                                                if row["guessedName"] != None else None, axis=1)
        return m_df

    # CHAT LOG ANALYSIS #
    def parse_chat(self, team_chat):
        """ public: given a chat json object, return a neat formatted df 
        of the chat transcriptions, minus helperbot """
        chat = pd.DataFrame(team_chat[0])  # [0] is bc weird nested list thing
        chat = chat[['user', 'nickname', 'message', 'time']]
        chat = chat.sort_values(by="time")
        chat = chat[chat['user'] != '100000000000000000000001']  # remove bot
        return chat

    def __get_msg_per(self, team):
        """ given a team series (row from t_df), return a series 
        listing the # of msgs per person in their respective columns """
        result = team.copy()
        chat = self.parse_chat(team['chat'])
        counts = chat['user'].value_counts()
        for user in counts.index:
            result[result == user] = counts[user]
        result = result[:-1]  # remove hanging chat column
        return result

    def __analyze_chat_team(self, t_df):
        """ given a row of t_df, get % of msgs per person in chat """
        msg_df = t_df.apply(self.__get_msg_per, axis=1)
        msg_df = msg_df.apply(lambda m: pd.to_numeric(m, errors="coerce")).astype(
            pd.Int32Dtype())  # convert to float
        msg_df = msg_df.fillna(0) # fill in missing users with 0
        msg_df = msg_df.apply(lambda r: 0 if r.sum() == 0 else r.div(r.sum()), axis=1)  # convert to %
        return msg_df

    def __analyze_chat_ind(self, u_df, t_df):
        """ get user-indexed % of msgs per person in chats """
        # reference msg_df and trace across u_df, filing in each corresponding cell of t_df
        msg_df = self.__analyze_chat_team(t_df)
        msgs = pd.DataFrame(index=u_df.index, columns=u_df.columns)
        for i in range(len(t_df.index)):
            round = t_df.index[i][0]
            for j in range(len(t_df.columns[:-1])):
                user = t_df.iloc[i, j]
                if (user != user):
                    continue  # to account for dropped users
                value = msg_df.iloc[i, j]
                msgs.loc[user, round] = value
        return msgs

    # PUBLIC ANALYSIS FUNCTIONS #
    def analyze(self, batch_id: str):
        """ public: for a batch_id, returns a BangDataResult obj """

        if self._verbose:
            print(">>> Analyzing batch " + batch_id)

        # setup
        batch_json = self.get_json(batch_id)

        if not self.__valid(batch_json):
            print(f"{batch_id} was not a complete batch. skipping this one.")
            return

        df = self.__batch_df(batch_json)
        t_df = self.__team_df(df)
        u_df = self.__user_df(df)

        # save to res obj
        res = BangDataResult(batch_id, batch_json, df, t_df, u_df)

        # analyze
        viability_team = self.__analyze_viability_team(u_df, t_df)
        viability_ind = self.__analyze_viability_ind(u_df)
        res.set("VIABILITY", viability_team, viability_ind)

        fracture_team = self.__analyze_fracture_team(u_df, t_df)
        fracture_ind = self.__analyze_fracture_ind(u_df)
        res.set("FRACTURE", fracture_team, fracture_ind)

        fracture_why_team = self.__analyze_fracture_why_team(u_df, t_df)
        fracture_why_ind = self.__analyze_fracture_why_ind(u_df)
        res.set("FRACTURE_WHY", fracture_why_team, fracture_why_ind)

        chat_team = self.__analyze_chat_team(t_df)
        chat_ind = self.__analyze_chat_ind(u_df, t_df)
        res.set("CHAT", chat_team, chat_ind)

        manipulation = self.__analyze_manipulation(batch_json)
        res.set("MANIPULATION", manipulation)

        # return
        return res

    def analyze_all(self, batches: [str]):
        """ public analyze many batch ids at once """
        res = list(filter(None, [self.analyze(batch) for batch in batches]))

        if self._verbose:
            print(f"\nAnalyzed the valid {len(res)} out of {len(batches)} batches.")

        return res

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
        self.labels = ["reconvene", "control"] if self.refPair1[0] == self.refPair2[0] else ["best", "worst"]
        
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

    def user_df(self):
        return self._user_df

    def viability(self, ind=False):
        """ returns the viability table, default at team-level, with added diff columns """
        viability =  self._analyses["VIABILITY_IND"] if ind else self._analyses["VIABILITY_TEAM"]
        if ind: return viability
        
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

### MULTIBATCH CLASS ###
class Multibatch():
    """ This class takes in a collection of BangDataRes objects and 
    summarizes their results, creating figures, doing analyses etc. 
    Whereas bangdatares operates at a batch level, this class operates
    at an experimental level. """

    def __init__(self, results: [BangDataResult], verbose=True):
        self._verbose = verbose
        self._raw_batches = results
        self._filt_batches = results
        self.summary = None
        self.viability_labels = results[0].labels # assume all are same toggle #427

        # default filters filter out nothing
        self._filters = []
        
        if verbose:
            print(f"Initialized Multibatch engine to analyze {len(results)} batches")

    ## MULTIBATCH SETTINGS ##
    def set_verbose(self, val):
        """ sets verbose flag to new value """
        self._verbose = bool(val)

        if self._verbose:
            print(f"verbose was set to {val}.")

    def set_batches(self, results: [BangDataResult]):
        """ sets the raw pool of batches to new col """
        old = self._raw_batches
        self._raw_batches = results

        if self._verbose:
            print("updated batches.")
            print(f"original # batches: {len(old)}, now: {len(results)}")

    ## FILTERING ##
    def set_filters(self, filters: [callable]):
        """ set the filter functions  
        the filtered functions provided should take in a bangdatares and return 
        bools true to keep and false to filter out 
        if bad inputs are passed in, will throw typeerrors """
        old_val = self._filters
        """ set the filter functions """
        self._filters = filters

        if self._verbose:
            print(f"filter functions were updated. Old list was {old_val}", end=" ")
            print(f". New list is {filters}")
    
    def filter(self):
        """ takes the batches saved in results and filters them
        through the set filter functions. throws error if filter fails,
        but filter rarely fails so more likely will just be wrong. """
        if self._verbose:
            print("\n>>> filtering batches. ensure that the functions take in a bangdatares obj and return a bool")

        filt = self._raw_batches
        for func in self._filters:
            try:
                filt = list(filter(func, filt))
            except: 
                print("Something went wrong. Set a new list of filter functions that take in a batchdatares and return a bool")
        
        self._filt_batches = filt

        if self._verbose:
            print("done filtering.")
            print(f"original # batches: {len(self._raw_batches)}, now: {len(self._filt_batches)}")

    ## SUMMARY TABLE ##
    def __batch_viabilities(self, batch: BangDataResult):
        """ extracts the viabilitys for the two refpairs, and the diffs between them """
        viability = batch.viability()

        # get scores for the first pair of refs and their diff
        r1 = viability.iloc[batch.refPair1[0]-1]['mean_viability'].item()
        r2 = viability.iloc[batch.refPair1[1]-1]['mean_viability'].item()
        rd = viability.iloc[batch.refPair1[0]-1][self.viability_labels[0]].item()

        # get scores for the second pair of refs and their diff
        d1 = viability.iloc[batch.refPair2[0]-1]['mean_viability'].item()
        d2 = viability.iloc[batch.refPair2[1]-1]['mean_viability'].item()
        dd = viability.iloc[batch.refPair2[0]-1][self.viability_labels[1]].item()

        return [r1, r2, rd, d1, d2, dd]
        
    def __batch_manipulations(self, batch: BangDataResult):
        """ extracts and calcs the expected and actual chances for manip """
        manip = batch.manipulation()
        act = manip.apply(lambda u: int(u['correct']) / u['numOptions'] \
            if u['numOptions'] is not None else None, axis=1).mean()
        exp = manip.apply(lambda u: 1 / u['numOptions'] \
            if u['numOptions'] is not None else None, axis=1).mean()

        return [act,exp]

    def summarize(self):
        """ prints a multibatch df that summarizes key results indexed by batch """
        if self._verbose:
            print(">>> Summarizing")
        
        summary = pd.DataFrame(columns=["batch", "initial_" + self.viability_labels[0], "later_" + self.viability_labels[0], \
            "diff_" + self.viability_labels[0], "initial_" + self.viability_labels[1], "later_" + self.viability_labels[1], \
            "diff_" + self.viability_labels[1], "manip_actual", "manip_chance", "refPair1", "refPair2"])
        i=1
        for batch in self._filt_batches:
            viability = self.__batch_viabilities(batch)
            manip = self.__batch_manipulations(batch)
            summary.loc[i] = [batch.batch, viability[0], viability[1], viability[2], viability[3], viability[4], viability[5], \
                manip[0], manip[1], batch.refPair1, batch.refPair2]
            i += 1
        self.summary = summary
        return summary

    def describe(self):
        """ describes all the columns in self.summary """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        if self._verbose:
            print(">>> Describing last summary\n")

        return self.summary.describe()

    ## ANALYSES ##
    def analyze_viability(self):
        """ wrapper function to set default viability analysis type based on whether 
        this is 1) rec/cont or 2) best/worst """
        if self.viability_labels == ['reconvene', 'control']:
            return self.analyze_viability_diff()
        else:
            return self.analyze_viability_raw()
            
    def analyze_viability_diff(self):
        """ performs all viability diff analyses across batches (section Rb)
        1. prints v2=R/B mean, std
        2. prints v2=D/W mean, std
        3. prints bar plot of v2=R/B, v2=D/W means + std error
        4. prints box plot of v2=R/B, v2=D/W 
        5. prints paired t-test results for diffs between v2=R/B and v2=D/W
        returns nothing """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        r = self.summary["diff_" + self.viability_labels[0]]
        d = self.summary["diff_" + self.viability_labels[1]]
        
        # 1. print r_diff mean, std
        print(f"\n>>> diff_{self.viability_labels[0]} mean, standard deviation:")
        print(f"n: {r.count()}, mean: {r.mean()}, std: {r.std()}")

        # 2. print d_diff mean, std
        print(f"\n>>> diff_{self.viability_labels[1]} mean, standard deviation:")
        print(f"n: {d.count()}, mean: {d.mean()}, std: {d.std()}")

        # 3. create barplot
        print("\n>>> barplot:")
        bar = plt.bar(np.arange(2), [r.mean(), d.mean()], yerr=[r.std(), d.std()], align='center')
        plt.title(f'Viability Diffs Between {self.viability_labels[0]} and {self.viability_labels[1]}')
        plt.xticks(np.arange(2), [self.viability_labels[0], self.viability_labels[1]])
        plt.xlabel('V2')
        plt.ylabel('Growth in Viability from Reference Round')
        plt.show()

        #4. create boxplot
        print("\n>>> boxplot:")
        box = plt.boxplot([r, d], positions=np.arange(2))
        plt.title(f'Viability Diffs Between {self.viability_labels[0]} and {self.viability_labels[1]}')
        plt.xticks(np.arange(2), [self.viability_labels[0], self.viability_labels[1]])
        plt.xlabel('V2')
        plt.ylabel('Growth in Viability from Reference Round')
        plt.show()

        # 5. paired t-test
        print(f"\n>>> paired t-test between diff_{self.viability_labels[0]} and diff_{self.viability_labels[1]}:")
        print(stats.ttest_rel(r, d))

    def analyze_viability_raw(self):
        """ performs all raw viability score analyses across batches (section Rb split)
        1. prints refPair1[0] mean, std
        2. prints refPair1[1] mean, std
        3. prints refPair2[0] mean, std
        4. prints refPair2[1] mean, std
        5. prints bar plot of v2=R/B, v2=D/W means + std error
        6. prints box plot of v2=R/B, v2=D/W 
        7. prints paired t-test results for initial scores R/B vs D/W
        8. prints paired t-test results for later scores R/B vs D/W  
        returns nothing """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        r1 = self.summary["initial_" + self.viability_labels[0]]
        r2 = self.summary["later_" + self.viability_labels[0]]
        d1 = self.summary["initial_" + self.viability_labels[1]]
        d2 = self.summary["later_" + self.viability_labels[1]]
        
        # 1. print r1 mean, std
        print(f"\n>>> initial_{self.viability_labels[0]} mean, standard deviation:")
        print(f"n: {r1.count()}, mean: {r1.mean()}, std: {r1.std()}")

        # 2. print r2 mean, std
        print(f"\n>>> later_{self.viability_labels[0]} mean, standard deviation:")
        print(f"n: {r2.count()}, mean: {r2.mean()}, std: {r2.std()}")

        # 3. print d1 mean, std
        print(f"\n>>> initial_{self.viability_labels[1]} mean, standard deviation:")
        print(f"n: {d1.count()}, mean: {d1.mean()}, std: {d1.std()}")

        # 4. print d2 mean, std
        print(f"\n>>> later_{self.viability_labels[1]} mean, standard deviation:")
        print(f"n: {d2.count()}, mean: {d2.mean()}, std: {d2.std()}")

        # 5. create barplot
        print("\n>>> barplot:")
        bar = plt.bar(np.arange(4), [r1.mean(), r2.mean(), d1.mean(), d2.mean()], yerr=[r1.std(), r2.std(), d1.std(), d2.std()], align='center')
        plt.title(f'Viabilities of {self.viability_labels[0]} and {self.viability_labels[1]}')
        plt.xticks(np.arange(4), ["Initial " + self.viability_labels[0],"Later " + self.viability_labels[0], \
            "Initial" + self.viability_labels[1], "Later" + self.viability_labels[1]])
        plt.xlabel('Round')
        plt.ylabel('Raw Viabilities')
        plt.plot([0,1], [r1.mean(), r2.mean()], c="r", lw=2)
        plt.plot([2,3], [d1.mean(), d2.mean()], c="r", lw=2)
        plt.show()

        # 6. create boxplot
        print("\n>>> boxplot:")
        box = plt.boxplot([r1, r2, d1, d2], positions=np.arange(4))
        plt.title(f'Viabilies of {self.viability_labels[0]} and {self.viability_labels[1]}')
        plt.xticks(np.arange(4), ["Initial " + self.viability_labels[0],"Later " + self.viability_labels[0], \
            "Initial" + self.viability_labels[1], "Later" + self.viability_labels[1]])
        plt.xlabel('Round')
        plt.ylabel('Raw Viabilities')
        plt.plot([0,1], [r1.median(), r2.median()], c="r", lw=2)
        plt.plot([2,3], [d1.median(), d2.median()], c="r", lw=2)
        plt.show()

        # 7. paired t-test initial
        print(f"\n>>> paired t-test between initial_{self.viability_labels[0]} and initial_{self.viability_labels[1]}:")
        print(stats.ttest_rel(r1, d1))

        # 8. paired t-test later
        print(f"\n>>> paired t-test between later_{self.viability_labels[0]} and later_{self.viability_labels[1]}:")
        print(stats.ttest_rel(r2, d2))

    def analyze_manipulation(self):
        """ performs all manipulation check analyses across batches (section Ra)
        1. prints manip_acutal mean, std
        2. prints manip_chancen mean, std
        3. prints plot with mean manip_chance bar + standard error, line for manip_actual
        4. prints paired t-test results for manip_acutal and manip_chance
        returns nothing """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        actual = self.summary['manip_actual']
        chance = self.summary['manip_chance']
        
        # 1. print manip_actual mean, std
        print("\n>>> manip_actual mean, standard deviation:")
        print(f"n: {actual.count()}, mean: {actual.mean()}, std: {actual.std()}")

        # 2. print manip_chance mean, std
        print("\n>>> manip_chance mean, standard deviation:")
        print(f"n: {chance.count()}, mean: {chance.mean()}, std: {chance.std()}")

        # 3. create barplot
        print("\n>>> barplot:")
        bar = plt.bar(np.arange(1), actual.mean(), yerr=actual.std(), align='center')
        plt.title('Manipulation Check Accuracy')
        plt.xticks(np.arange(1), '')
        plt.xlabel('Actual Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(bottom=0, top=chance.mean()+0.1)
        plt.axhline(y=chance.mean(),linewidth=2,label="Chance Accuracy") #threshold line
        plt.legend()
        plt.show()

        # 4. paired t-test
        print("\n>>> paired t-test between manip_acutal and manip_chance:")
        print(stats.ttest_rel(actual, chance))
