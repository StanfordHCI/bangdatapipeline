# bangdatapipeline
a module for the data pipeline for the Scaled Humanity team

questions: slack i-gao

## how bang exports data
Bang exports each batch as a long JSON file that is basically an exact transcript of the redux store. This data pipeline seeks to abstract away the parsing of that JSON data.

## import
Usage Instructions:
1. clone this repo
2. install the package
3. import in desired classes

```
# Install bangdatapipeline & multibatch packages
!pip install ./bangdatapipeline

from bangdatapipeline import BangDataPipeline
from multibatch import Multibatch # imports default Multibatch
from multibatch.parallelworlds import Multibatch # or whatever version you add
```

## class 1: BangDataPipeline
The BangDataPipeline class in the `bangdata.py` file handles JSON retrieval and shaping. The current structure assumes that each round is followed by a mid-survey with the following kinds of questions:

* **Viability** -- a set of multiple-choice questions whose numerical values we want to *average.*
* **Fracture** -- a set of two questions, one Yes/No, one short response, that says whether a participant wants to keep their team and why.

If you want to add in analysis of other types of questions, implement a new class in a new file that inherits the base BangDataPipeline and overload functions. Import as `from bangdatapipeline.FILENAME import CLASSNAME`.

Because Bang does not label questions in the JSON file, the BangDataPipeline class requires that you specify what 0-index these questions fall on. For example, if my survey has 16 questions, the first 14 of which are viability questions, I'd need to specify a SETTINGS object and initialize the BangDataPipleline with those settings. By default, BangDataPipeline is initialized with a simple setting of survey length = 0.

```
SETTINGS = {
    "VIABILITY_START": 0, 
    "VIABILITY_END": 13,
    "FRACTURE_INDEX": 14,
    "FRACTURE_WHY": 15,
    "LENGTH": 16
}
bdp = BangDataPipeline(TOKEN, SETTINGS) # TOKEN is your Bang API token
```

BangDataPipeline includes a set of functions to analyze a batch. Given a batch ID, BangDataPipeline will fetch the batch json and create tables respresenting viability scores, fracture results, etc. As an example, see the code block below:

```
# programmatically select the 5 most recent batches
fetch = bdp.fetch() # returns a df
batches = fetch[:5]['batch_id'].tolist() # select the 5 IDs you want

singleres = bdp.analyze(batches[0]) # analyze one ID 
res = bdp.analyze_all(batches) # loop and analyze all in the list
```

## class 2: BangDataResult
You should never need to directly import this class, but you will need to interact with it. When BangDataPipeline analyzes a batch, it returns a BangDataResult class that acts as the Viewer for the individual batch's results.

Some useful fields & functions (here, `res` is of the BangDataResult class)
* `res.batch` = batch ID
* `res.users` = a list of user IDs in batch
* `res.teams` = a list of team IDs
* `res.refPair1` = a tuple of the experimental rounds (expRound1 in JSON) 
* `res.refPair2` = a tuple of the second experimental rounds (expRound2 in JSON)
* `res.labels` = what the two experimental rounds are called
* `parse_chat()` = given a JSON snippet of a chat log, formats a nice table of time-ordered chat log
* `combine()` = given two tables, zips each entry to form a table where cell i,j is (A[i,j], B[i,j])
* `json()` = batch's raw JSON
* `raw_df()` = base df for analysis
* `team_df()` = df indexed by teams with member IDs as entries
* `raw_chats()` = series of chats indexed by team
* `user_df()` = df indexed by user with mid-survey JSONs named by round
* `viability()` = user-indexed viability df
* `fracture()` = user-indexed fracture outcome df
* `manipulation()` = user-indexed manipulation check answers df

## class 3: Multibatch
Multibatch is an engine to aggregate multiple batch results into summary analyses on a study level. The base class is in `multibatch/base.py`; an example of a study-specific child implementation is in `multibatch/parallelworlds.py`. 


