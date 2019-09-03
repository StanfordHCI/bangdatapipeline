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
            if batch['status'] == "completed":
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

        return sum_viability

    def __analyze_viability_team(self, u_df, t_df):
        """ get team-indexed viability scores with average and diff """
        viability = t_df.apply(lambda team: team[:-1].apply(lambda userid: self.__ind_viability(
            self.__get_survey(u_df, team.name[0], userid)) if(userid == userid) else None), axis=1)
        # translation: across rows of t_df minus chat (so user ids), use the id and the round number (team.name[0]) to query ind_viability

        # mean
        viability['mean_viability'] = viability.mean(axis=1)

        # get differences
        final_viability = viability.tail(1)['mean_viability'].item()
        viability['re_minus_ref'] = - \
            (viability['mean_viability'] - final_viability)

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
        m_df = pd.DataFrame(columns=['user', 'guessedName', 'actualName'])
        i = 1
        for user in raw['users']:
            user_id = user['user']['_id']
            if('survey' not in user or 'singleTeamQuestion' not in user['survey']):
                m_df.loc[i] = [user_id, None, None]
            else:
                guessed = user['survey']['singleTeamQuestion']['chosenPartnerName']
                actual = user['survey']['singleTeamQuestion']['actualPartnerName']
                m_df.loc[i] = [user_id, guessed, actual]
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
        msg_df = msg_df.apply(lambda r: r.div(r.sum()), axis=1)  # convert to %
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
        return [self.analyze(batch) for batch in batches]

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
        self.expRounds = json['expRounds']

        ## PRIVATE FIELDS ##
        self.__json = json
        self.__raw_df = df
        self.__team_df = t_df
        self.__user_df = u_df
        self.__analyses = {}

    ## SETTERS ##
    def set(self, key: str, team, ind=None):
        """ ideally not a public method """
        self.__analyses[key + "_TEAM"] = team
        if (ind is not None):
            self.__analyses[key + "_IND"] = ind

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
        return self.__json

    def raw_df(self):
        return self.__raw_df

    def team_df(self):
        return self.__team_df['user']

    def user_df(self):
        return self.__user_df

    def viability(self, ind=False):
        """ returns the viability table, default at team-level """
        return self.__analyses["VIABILITY_IND"] if ind else self.__analyses["VIABILITY_TEAM"]
    
    def fracture(self, ind=False):
        """ returns the fracture table, default at team-level """
        return self.__analyses["FRACTURE_IND"] if ind else self.__analyses["FRACTURE_TEAM"]
    
    def fracture_why(self, ind=False):
        """ returns the fracture table, default at team-level """
        return self.__analyses["FRACTURE_WHY_IND"] if ind else self.__analyses["FRACTURE_WHY_TEAM"]
    
    def chat(self, ind=False):
        """ returns the chat composition table, default at team-level """
        return self.__analyses["CHAT_IND"] if ind else self.__analyses["CHAT_TEAM"]
    
    def manipulation(self):
        return self.__analyses["MANIPULATION_TEAM"]

    def display(self):
        return list(self.__analyses.values())
    
    def __str__(self):
        return str(list(self.__analyses.values()))
