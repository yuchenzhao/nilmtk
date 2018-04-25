from __future__ import print_function, division
import itertools
from copy import deepcopy
from collections import OrderedDict
from warnings import warn

import nilmtk
import pandas as pd
import numpy as np
from hmmlearn import hmm

from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator

# Python 2/3 compatibility
from six import iteritems
from builtins import range

SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)


def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """Sorts the transition matrix according to increasing order of
    power means; as returned by mapping

    Parameters
    ----------
    mapping :
    A : numpy.array of shape (k, k)
        transition matrix
    """
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]


def compute_A_fhmm(list_A):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    --------
    result : Combined Pi for the FHMM
    """
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_means_fhmm(list_means):
    """
    Returns
    -------
    [mu, cov]
    """
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]


def compute_pi_fhmm(list_pi):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    -------
    result : Combined Pi for the FHMM
    """
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result


def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

    combined_model = hmm.GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    
    return combined_model


def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping


def decode_hmm(length_sequence, centroids, appliance_list, states):
    """
    Decodes the HMM state sequence
    """
    hmm_states = {}
    hmm_power = {}
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):

        factor = total_num_combinations
        for appliance in appliance_list:
            # assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]
    return [hmm_states, hmm_power]


class FHMM(Disaggregator):
    """
    Attributes
    ----------
    model : dict
    predictions : pd.DataFrame()
    meters : list
    MIN_CHUNK_LENGTH : int
    """

    def __init__(self):
        self.model = {}
        self.predictions = pd.DataFrame()
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'FHMM'



    def train_across_buildings(self, ds, list_of_buildings, list_of_appliances,
              min_activation=0.05, **load_kwargs):

        """

        :param ds: nilmtk.Dataset
        :param list_of_buildings: List of buildings to use for training
        :param list_of_appliances: List of appliances (nilm-metadata names)
        :param min_activation: Minimum activation (in fraction) to use a home in training
        :param load_kwargs:
        :return:
        """
        self.list_of_appliances = list_of_appliances
        models = {}

        for appliance in list_of_appliances:
            print("Training for", appliance)
            o = []
            for building_num in list_of_buildings:

                building = ds.buildings[building_num]
                elec = building.elec
                try:
                    df = next(elec[appliance].load(**load_kwargs)).squeeze()
                    appl_power = df.dropna().values.reshape(-1, 1)
                    activation = (df > 10).sum() * 1.0 / len(df)
                    if activation > min_activation:
                        o.append(appl_power)
                except:
                    pass

            if len(o) > 1:
                o = np.array(o)
                mod = hmm.GaussianHMM(2, "full")
                mod.fit(o)
                models[appliance] = mod
                print("Means for %s are" % appliance)
                print(mod.means_)
            else:
                print("Not enough samples for %s" % appliance)

        new_learnt_models = OrderedDict()
        for appliance, appliance_model in iteritems(models):
            startprob, means, covars, transmat = sort_learnt_parameters(
                appliance_model.startprob_, appliance_model.means_,
                appliance_model.covars_, appliance_model.transmat_)
            new_learnt_models[appliance] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
            new_learnt_models[appliance].means_ = means
            new_learnt_models[appliance].covars_ = covars

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined
        self.meters = [nilmtk.global_meter_group.select_using_appliances(type=appliance).meters[0]
                       for appliance in self.individual.iterkeys()]

    def train(self, metergroup, num_states_dict={}, **load_kwargs):
        """Train using 1d FHMM.

        Places the learnt model in `model` attribute
        The current version performs training ONLY on the first chunk.
        Online HMMs are welcome if someone can contribute :)
        Assumes all pre-processing has been done.
        """
        learnt_model = OrderedDict()
        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for i, meter in enumerate(metergroup.submeters().meters):
            power_series = meter.power_series(**load_kwargs)
            meter_data = next(power_series).dropna()
            X = meter_data.values.reshape((-1, 1))
            assert X.ndim == 2
            self.X = X

            if num_states_dict.get(meter) is not None:
                # User has specified the number of states for this appliance
                num_total_states = num_states_dict.get(meter)

            else:
                # Find the optimum number of states
                states = cluster(meter_data, max_num_clusters)
                num_total_states = len(states)

            print("Training model for submeter '{}'".format(meter))
            learnt_model[meter] = hmm.GaussianHMM(num_total_states, "full")

            # Fit
            learnt_model[meter].fit(X)

            # Check to see if there are any more chunks.
            # TODO handle multiple chunks per appliance.
            try:
                next(power_series)
            except StopIteration:
                pass
            else:
                warn("The current implementation of FHMM"
                     " can only handle a single chunk.  But there are multiple"
                     " chunks available.  So have only trained on the"
                     " first chunk!")

        # Combining to make a AFHMM
        self.meters = []
        new_learnt_models = OrderedDict()
        for meter in learnt_model:
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[meter].startprob_, learnt_model[meter].means_,
                learnt_model[meter].covars_, learnt_model[meter].transmat_)
                
            new_learnt_models[meter] = hmm.GaussianHMM(startprob.size, "full")
            new_learnt_models[meter].startprob_ = startprob
            new_learnt_models[meter].transmat_ = transmat
            new_learnt_models[meter].means_ = means
            new_learnt_models[meter].covars_ = covars
            # UGLY! But works.
            self.meters.append(meter)

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined

    def disaggregate_chunk(self, test_mains):
        """Disaggregate the test data according to the model learnt previously

        Performs 1D FHMM disaggregation.

        For now assuming there is no missing data at this stage.
        """
        # See v0.1 code
        # for ideas of how to handle missing data in this code if needs be.

        # Array of learnt states
        learnt_states_array = []
        test_mains = test_mains.dropna()
        length = len(test_mains.index)
        temp = test_mains.values.reshape(length, 1)
        learnt_states_array.append(self.model.predict(temp))
        # Model
        means = OrderedDict()
        for elec_meter, model in iteritems(self.individual):
            means[elec_meter] = (
                model.means_.round().astype(int).flatten().tolist())
            means[elec_meter].sort()

        decoded_power_array = []
        decoded_states_array = []

        for learnt_states in learnt_states_array:
            [decoded_states, decoded_power] = decode_hmm(
                len(learnt_states), means, means.keys(), learnt_states)
            decoded_states_array.append(decoded_states)
            decoded_power_array.append(decoded_power)

        prediction = pd.DataFrame(
            decoded_power_array[0], index=test_mains.index)

        return prediction


    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        sample_period : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        import warnings
        warnings.filterwarnings("ignore", category=Warning)

        for chunk in mains.power_series(**load_kwargs):

            # Check that chunk is sensible size before resampling
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            # Start disaggregation
            predictions = self.disaggregate_chunk(chunk)
            for meter in predictions.columns:

                meter_instance = meter.instance()
                cols = pd.MultiIndex.from_tuples([chunk.name])
                predicted_power = predictions[[meter]]
                if len(predicted_power) == 0:
                    continue
                data_is_available = True
                output_df = pd.DataFrame(predicted_power)
                output_df.columns = pd.MultiIndex.from_tuples([chunk.name])
                key = '{}/elec/meter{}'.format(building_path, meter_instance)
                output_datastore.append(key, output_df)

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=self.meters
            )

    def disaggregate_across_buildings(self, ds, output_datastore, list_of_buildings, **load_kwargs):
        """

        :param ds:
        :param list_of_buildings:
        :return:
        """

        def get_meter_instance(ds, building_num, appliance):
            elec = ds.buildings[building_num].elec
            meters = elec.submeters().meters
            for meter in meters:
                if meter.appliances[0].type['type'] == appliance:
                    return meter.instance()
            return -1

        for building in list_of_buildings:
            print("Disaggregating for building %d" % building)
            mains = ds.buildings[building].elec.mains()
            load_kwargs = self._pre_disaggregation_checks(load_kwargs)

            load_kwargs.setdefault('sample_period', 60)
            load_kwargs.setdefault('sections', mains.good_sections())

            timeframes = []
            building_path = '/building{}'.format(mains.building())
            mains_data_location = building_path + '/elec/meter1'
            data_is_available = False

            import warnings
            warnings.filterwarnings("ignore", category=Warning)
            building_elec = ds.buildings[building].elec
            self.meters = []
            for appliance in self.list_of_appliances:
                m_instance = get_meter_instance(ds, building, appliance)
                if m_instance != -1:
                    self.meters.append(building_elec[m_instance])
                else:
                    pass

            for chunk in mains.power_series(**load_kwargs):
                # Check that chunk is sensible size before resampling
                if len(chunk) < self.MIN_CHUNK_LENGTH:
                    continue

                # Record metadata
                timeframes.append(chunk.timeframe)
                measurement = chunk.name

                # Start disaggregation
                predictions = self.disaggregate_chunk(chunk)
                for meter in predictions.columns:

                    if type(meter) is str:
                        # training done across homes
                        meter_instance = get_meter_instance(ds, building, meter)
                        if meter_instance == -1:
                            continue
                    else:
                        meter_instance = meter.instance()
                    cols = pd.MultiIndex.from_tuples([chunk.name])
                    predicted_power = predictions[[meter]]
                    if len(predicted_power) == 0:
                        continue
                    data_is_available = True
                    output_df = pd.DataFrame(predicted_power)
                    output_df.columns = pd.MultiIndex.from_tuples([chunk.name])
                    key = '{}/elec/meter{}'.format(building_path, meter_instance)
                    output_datastore.append(key, output_df)

                # Copy mains data to disag output
                output_datastore.append(key=mains_data_location,
                                        value=pd.DataFrame(chunk, columns=cols, dtype='float32'))

            if data_is_available:
                self._save_metadata_for_disaggregation(
                    output_datastore=output_datastore,
                    sample_period=load_kwargs['sample_period'],
                    measurement=measurement,
                    timeframes=timeframes,
                    building=mains.building(),
                    meters=self.meters
                )

    ###################### Methods Below by Yuchen ############################

    def output_probability(self, test_elec, file_dist_output):
        for i, chunk in enumerate(test_elec.mains().load(sample_period = 60)):
            test_mains = chunk.dropna()
            length = len(test_mains.index)
            temp = test_mains.values.reshape(length, 1)
            np.save(file_dist_output + "_main_" + str(i) + ".npy",
                    self.model.predict_proba(temp))


    def output_states(self, used_meters, test_elec, file_states_output):
        main_meters = test_elec.mains()
        sub_meters = test_elec.submeters()
        pred = {}
        gt = {}
        for i, chunk in enumerate(main_meters.load(sample_period=60)):
            chunk_drop_na = chunk.dropna()
            pred[i] = self.disaggregate_chunk_with_states(chunk_drop_na)
            gt[i] = {}

            for meter in test_elec.submeters().meters:
                if meter not in used_meters:
                    continue
                gt[i][meter] = next(meter.load(sample=60)).dropna()
                gt[i][meter] = add_states_column(meter, gt[i][meter]) 



    def add_states_column(self, meter, df_power):
        for elec_meter, model in iteritems(slef.individual):
            if elec_meter == meter:
                means = model.means_.round().astype(int).flatten().tolist()
                means.sort()
        states = []
        ## TODO: convert the power values to states. Find the closest mean value to a power value, and the index of the mean value is the state.


    def disaggregate_chunk_with_states(self, test_mains):
        learnt_states_array = []
        test_mains = test_mains.dropna()
        length = len(tst_mains.index)
        temp = test_mains.values.reshape(length, 1)
        learnt_states_array.append(self.model.predict(temp))

        means = OrderedDict()
        for elec_meter, model in iteritems(self.individual):
            means[elec_meter] = model.means_.round().astype(int).flatten().tolist())
            means[elec_meter].sort()

        decoded_power_array = []
        decoded_states_array = []

        for learnt_states in learn_states_array:
            decoded_states, decoded_power = decode_hmm(len(learnt_states), means, means.keys(), learnt_states)
            decoded_states_array.append(decoded_states)
            decoded_power_array.append(decoded_power)

        prediction = pd.DataFrame(decoded_power_array[0], decoded_states_array[0], index=test_mains.index)

        return prediction

    def output_for_metrics(self, used_meters, test_elec, file_dist_output, file_states_output):
        output_probability(test_elec, file_dist_output)
        output_states(used_meters, test_elec, file_states_output)
