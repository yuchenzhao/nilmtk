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

for i, chunk in enumerate(test_elec.mains().load(sample_period=60)):
    chunk_drop_na = chunk.dropna()
    start = time.time()
    print("Disaggregation gets started now")
    fhmm.output_for_metrics(chunk_drop_na, "data/output/dist.npy")
    end = time.time()
    print("Disaggregation done. Runtime = ", end - start, " seconds.")

