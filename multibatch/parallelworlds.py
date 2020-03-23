"""parallelWorlds.py: Bang data processing module
********************************************
Author: Irena & Junior
This module encapsulates the analysis engine for the Parallel Worlds project
"""

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
import textwrap
import researchpy

from bangdatapipeline.bangdataresult import BangDataResult
from multibatch.base import Multibatch as Base
from multibatch.style import style_plt

__version__ = "0.1.0"


def label_diff(i,j,text,tops):
    """ helper function to plot the diff between two bars 
        i: the initial 0-index bar to start at
        j: the 0-index bar to end at
        text: text to put on top of annotation
        tops: the maxs of all data plotted """
    X = np.arange(len(tops))
    x = (X[i]+X[j])/2
    y = max(tops[i], tops[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                        'shrinkA':20,'shrinkB':20,'linewidth':1.2}
    text_offset = 8*abs(i-j)
    plt.annotate(text, xy=(X[i],y+text_offset), zorder=10)
    plt.annotate('', xy=(X[i]+0.05,y), xytext=(X[j]-0.05,y), arrowprops=props)


### MULTIBATCH CLASS ###
class Multibatch(Base):
    """ This class takes in a collection of BangDataRes objects and 
    summarizes their results, creating figures, doing analyses etc. 
    Whereas bangdatares operates at a batch level, this class operates
    at an experimental level. """

    def __init__(self, results: [BangDataResult], style_fn=style_plt, verbose=True):
        Base.__init__(self, results, verbose)
        
        # init styles
        style_fn()

        self.summary = None

    ## SUMMARY TABLE ##
    def __batch_viabilities(self, batch: BangDataResult, block):
        """ extracts the viabilitys for the two refpairs, and the diffs between them """
        viability = batch.viability(block=block)

        # get scores for the first pair of refs and their diff
        r1 = viability.iloc[batch.refPair1[0]-1]['mean_viability'].iloc[0]
        r2 = viability.iloc[batch.refPair1[1]-1]['mean_viability'].iloc[0]
       
        # get scores for the second pair of refs and their diff
        d1 = viability.iloc[batch.refPair2[0]-1]['mean_viability'].iloc[0]
        d2 = viability.iloc[batch.refPair2[1]-1]['mean_viability'].iloc[0]
       
        return [r1, r2, d1, d2]

    def __batch_viability(self, batch: BangDataResult):
        """ extracts the average viability of the initial non refpair (so not last two) rounds """
        viability = batch.viability()
        scores = viability.iloc[0 : batch.numRounds]['mean_viability']
        return scores
        
    def __batch_manipulations(self, batch: BangDataResult):
        """ extracts and calcs the expected and actual chances for manip """
        manip = batch.manipulation()
        exp = manip.apply(lambda u: 1 / u['numOptions'] \
            if u['numOptions'] is not None else None, axis=1).mean()
        act = (manip['correct'].value_counts()[True] \
            if True in manip['correct'].value_counts() else 0) / manip['correct'].value_counts().sum()

        return [act,exp]

    def __batch_users(self, batch: BangDataResult):
        """ extracts the number of final active users in a batch """
        viab = batch.viability()
        return viab.tail(1)['user'].count().sum()

    def summarize(self, block=False):
        """ prints a multibatch df that summarizes key results indexed by batch """
        if self._verbose:
            print(">>> Summarizing")

        summary = pd.DataFrame(columns=["batch", *(np.arange(self._filt_batches[0].numRounds) + 1), "median", "early_mean", "initial_" + self.viability_labels[0], "later_" + self.viability_labels[0], \
            "initial_" + self.viability_labels[1], "later_" + self.viability_labels[1], \
            "manip_actual", "manip_chance", "refPair1", "refPair2", "numUsers"])

        i=1
        for batch in self._filt_batches:
            rounds = self.__batch_viability(batch)
            viability = self.__batch_viabilities(batch, block)
            median_candidates = [batch.refPair1[1]-1, batch.refPair2[1]-1] if (self.viability_labels[1] == "control") else [batch.refPair1[1]-1]
            median = np.median(rounds.drop(rounds.index[median_candidates])) 
            mean = np.mean(rounds[:batch.numRounds - 2])
            manip = self.__batch_manipulations(batch)
            num_users = self.__batch_users(batch)
            summary.loc[i] = [batch.batch, *rounds, median, mean, viability[0], viability[1], viability[2], viability[3], \
                manip[0], manip[1], batch.refPair1, batch.refPair2, num_users]
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
    def analyze_viability(self, *diff_args: [int, int, str]):
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

        title="Viability Scores of Rounds"
        labels=[textwrap.fill(text, 12) for text in \
            ["Initial " + self.viability_labels[0],"Later " + self.viability_labels[0], \
            "Initial " + self.viability_labels[1], "Later " + self.viability_labels[1]]]
        ylabel = "Team Mean Viability"

        order = [r1, r2, d1, d2]
        means = [x.mean() for x in order]
        stds = [x.std() for x in order]
        maxs = [x.max()  for x in order]

        # 5. create barplot
        print("\n>>> barplot:")
        plt.figure(1)
        bar = plt.bar(np.arange(4), means, yerr=stds, align='center')
        plt.title(title)
        plt.xticks(np.arange(4), labels)
        plt.ylabel(ylabel)
        plt.savefig("raw_bar.pdf")
        plt.show()

        # 6. create boxplot
        print("\n>>> boxplot:")
        plt.figure(2)
        box = plt.boxplot(order, positions=np.arange(4))
        plt.title(title)
        plt.xticks(np.arange(4), labels)
        plt.ylabel(ylabel)
        plt.ylim(top=70, bottom=14)
        
        # label diffs
        for args in diff_args:
            label_diff(*args, maxs)  
        

        plt.savefig("raw_box.pdf")
        plt.show()

        # 7. paired t-test initial
        print(f"\n>>> paired t-test between initial_{self.viability_labels[0]} and initial_{self.viability_labels[1]}:")
        tt1 = researchpy.ttest(r1, d1, paired=True)[1]
        print(tt1)

        # 8. paired t-test later
        print(f"\n>>> paired t-test between later_{self.viability_labels[0]} and later_{self.viability_labels[1]}:")
        tt1 = researchpy.ttest(r2, d2, paired=True)[1]
        print(tt1)
            
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

        title=f'Viability Diffs Between {self.viability_labels[0]} and {self.viability_labels[1]}'
        labels=[textwrap.fill(text, 12) for text in \
            [self.viability_labels[0], self.viability_labels[1]]]
        ylabel = "Growth in Viability from Reference Round"
        
        order = [r, d]
        means = [x.mean() for x in order]
        stds = [x.std() for x in order]
        maxs = [x.max()  for x in order]

        # 3. create barplot
        print("\n>>> barplot:")
        plt.figure(1)
        bar = plt.bar(np.arange(2), means, yerr=stds, align='center')
        plt.title(title)
        plt.xticks(np.arange(2), labels)
        #plt.xlabel('V2')
        plt.ylabel(ylabel)
        plt.savefig("diff_bar.pdf")
        plt.show()

        #4. create boxplot
        print("\n>>> boxplot:")
        plt.figure(2)
        box = plt.boxplot(order, positions=np.arange(2))
        plt.title(title)
        plt.xticks(np.arange(2), labels)
        plt.xlabel('V2')
        plt.ylabel(ylabel)
        plt.savefig("diff_box.pdf")
        plt.show()

        # 5. paired t-test
        print(f"\n>>> paired t-test between diff_{self.viability_labels[0]} and diff_{self.viability_labels[1]}:")
        # print(stats.ttest_rel(r, d))
        tt1 = researchpy.ttest(r, d, paired=True)[1]
        print(tt1)

    def analyze_viability_early(self, *diff_args: [int, int, str]):
        """ performs analyze_viability but with early viability score (average of initial)
        really just a study 1 figure generator
        1. prints refPair1[0] mean, std
        2. prints refPair1[1] mean, std
        3. prints median mean, std
        4. prints bar plot
        5. prints box plot 
        6. prints paired t-test results for refPair1[0] and refPair1[1] 
        7. prints paired t-test results for median and refPair1[1]
        returns nothing """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        r1 = self.summary["initial_" + self.viability_labels[0]]
        r2 = self.summary["later_" + self.viability_labels[0]]
        m = self.summary["median"]
        
        # 1. print r1 mean, std
        print(f"\n>>> initial_{self.viability_labels[0]} mean, standard deviation:")
        print(f"n: {r1.count()}, mean: {r1.mean()}, std: {r1.std()}")

        # 2. print r2 mean, std
        print(f"\n>>> later_{self.viability_labels[0]} mean, standard deviation:")
        print(f"n: {r2.count()}, mean: {r2.mean()}, std: {r2.std()}")

        # 3. print m mean, std
        print(f"\n>>> median round mean, standard deviation:")
        print(f"n: {m.count()}, mean: {m.mean()}, std: {m.std()}")

        title="Viability Scores of Rounds"
        labels=[textwrap.fill(text, 12) for text in \
            ["Best Initial Round","Reconvened Round","Median Round"]]
        ylabel = "Team Mean Viability"
        
        order = [r1, r2, m]
        means = [x.mean() for x in order]
        stds = [x.std() for x in order]
        maxs = [x.max() for x in order]

        # 4. create barplot
        print("\n>>> barplot:")
        plt.figure(1)
        bar = plt.bar(np.arange(3), means, yerr=stds, align='center')
        plt.title(title)
        plt.xticks(np.arange(3), labels)
        plt.ylabel(ylabel)
        plt.savefig("early_bar.pdf")
        plt.show()
        
        # 5. create boxplot
        print("\n>>> boxplot:")
        plt.figure(2)
        box = plt.boxplot(order, positions=np.arange(3))
        plt.title(title)
        plt.xticks(np.arange(3), labels)
        plt.ylabel(ylabel)
        plt.ylim(top=70, bottom=14)  

        # label diffs
        for args in diff_args:
            label_diff(*args, maxs)  

        plt.savefig("early_box.pdf")
        plt.show()     

        # 6. paired t-test r1 and r2
        print(f"\n>>> paired t-test between median and reconvened {self.viability_labels[0]}:")
        # print(stats.ttest_rel(r2, e))
        tt1 = researchpy.ttest(r1, r2, paired=True)[1]
        print(tt1)

        # 7. paired t-test median and r2
        print(f"\n>>> paired t-test between median and reconvened {self.viability_labels[0]}:")
        # print(stats.ttest_rel(r2, e))
        tt1 = researchpy.ttest(m, r2, paired=True)[1]
        print(tt1)

        # return data for exterior processing
        return order

    def analyze_viability_all(self):
        """ boxplots all round viabilities just by raw round # """
        if self.df is None:
            self.aggregate()

        boxplot = self.df.boxplot()
        return boxplot

    def analyze_viability_single(self):
        """ boxplots the value later reconvene - later median """
        # error checking
        if self.summary is None:
            print("You must run .summarize() before running this function")

        r = self.summary["later_" + self.viability_labels[0]]
        d = self.summary["later_" + self.viability_labels[1]]
        s = r-d
        
        # 1. print r1 mean, std
        print(f"\n>>> single value later {self.viability_labels[0]} - {self.viability_labels[1]} mean, standard deviation:")
        print(f"n: {s.count()}, mean: {s.mean()}, std: {s.std()}")

        # 2. describe
        print(">>> single value desc")
        print(s.describe())

        # 3. boxplot
        print("\n>>> boxplot:")
        box = plt.boxplot(s, positions=np.arange(1))
        plt.title(f'Single value later {self.viability_labels[0]} - {self.viability_labels[1]}')
        plt.xticks(np.arange(1), ["Single Value"])
        plt.xlabel('')
        plt.ylabel('Raw Viabilities')
        plt.savefig("single_box.pdf")
        plt.show()

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
        plt.figure(figsize=[7.13, 2])
        bar = plt.barh(np.arange(1), actual.mean(), align='center', edgecolor="black", height=0.5)
        plt.errorbar(actual.mean(), np.arange(1), xerr=actual.std(), ecolor="black")
        plt.title('Manipulation Check Accuracy')
        plt.yticks(np.arange(1), '')
        plt.ylabel('Actual Accuracy')
        plt.xlabel('Accuracy')
        plt.xlim(left=0, right=1.0)
        plt.axvline(x=chance.mean(),linewidth=4,color="red",label="Chance Accuracy") #threshold line
        plt.legend()
        plt.tight_layout()
        plt.savefig("manip.pdf", bbox_inches = "tight")
        plt.show()

        # 4. paired t-test
        print("\n>>> paired t-test between manip_acutal and manip_chance:")
        # print(stats.ttest_rel(actual, chance))
        tt1 = researchpy.ttest(actual, chance, paired=True)[1]
        print(tt1)
