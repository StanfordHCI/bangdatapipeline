"""multibatch.py: Bang data processing module
********************************************
Author: Irena & Junior
This module encapsulates the basic multibatch analysis engine
Other files in the multibatch folder are children of this class
"""

import pandas as pd

from bangdatapipeline.bangdataresult import BangDataResult

__version__ = "1.0.0"

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
        self.df = None
        self.chats = None
        self.viability_labels = results[0].labels # assume all are same toggle #427

        # default filters filter out nothing
        self._filters = []
        
        if verbose:
            print(f"Initialized Multibatch engine to analyze {len(results)} batches")

    def batches(self):
        """ returns analyzed list of batch bangdatares """
        return self._filt_batches

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

    ## MAIN TABLE ##
    def aggregate(self):
        """ batch-indexed viability table with refPair data """
        num_rounds = self._filt_batches[0].numRounds
        table = pd.DataFrame(columns=["batch", *list(range(1, num_rounds+1)), *self.viability_labels])
        i=1
        for batch in self._filt_batches:
            viability = batch.viability()['mean_viability']
            table.loc[i] = [batch.batch, *viability, batch.refPair1, batch.refPair2]
            i += 1
        self.df = table
        return(table)

    ## CHAT TABLE ##
    def get_chats(self):
        """ batch-indexed viability table with refPair data """
        num_rounds = self._filt_batches[0].numRounds
        table = pd.DataFrame(columns=["batch", *list(range(1, num_rounds+1)), *self.viability_labels])
        i=1
        for batch in self._filt_batches:
            chats = batch.raw_chats()
            table.loc[i] = [batch.batch, *chats, batch.refPair1, batch.refPair2]
            i += 1
        self.chats = table
        return(table)