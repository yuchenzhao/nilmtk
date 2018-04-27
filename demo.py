from __future__ import print_function, division
import time
import pandas as pd
import numpy as np
from six import iteritems
import warnings
from nilmtk.disaggregate import fhmm_exact

warnings.filterwarnings("ignore")

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation

train = DataSet("data/redd.h5")
test = DataSet("data/redd.h5")

train.set_window(end="4-30-2011")
train.set_window(start="4-30-2011")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

top_5_train_elec = train_elec.submeters().select_top_k(k=5)

fhmm = fhmm_exact.FHMM()
fhmm.train(top_5_train_elec, sample_period=60)

pred = {}
gt = {}

start = time.time()
print("Disaggregation gets started now")
fhmm.output_for_metrics(top_5_train_elec.submeters().meters,
                        test_elec,
                        60,
                        "data/output/dist",
                        "data/output/state")
end = time.time()
print("Disaggregation done. Runtime = ", end - start, " seconds.")

