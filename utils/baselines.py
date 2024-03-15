from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

def fit_BN_model(data): 
    """
    Fit Bayesian network to data
    Follow GT structure: season -> pneu, season -> inf, pneu -> dysp, pneu -> cough, inf -> cough, inf -> nasal 

    the rows with NaNs for the symptoms should still contribute to training of the CPDs for season, pneu and inf. 
    but in the standard pgmpy fit method, the rows with empty values are simply skipped over 
    we should be able to use Expectation Maximization to deal with this, but this does not work in pgmpy
    since the missing values are always leaf nodes, we can solve the problem by splitting 
    the BN in a top and a bottom part, and training them separately with the full dataset.
    """ 

    # top part: train CPDs for pneu, inf and season
    state_names = {"pneu": ["yes", "no"], "inf": ["yes", "no"], "season": ["warm", "cold"]}
    model_top = BayesianNetwork([("season", "pneu"), ("season", "inf")])
    model_top.fit(data[["pneu", "inf", "season"]], estimator=BayesianEstimator, prior_type="K2", state_names=state_names)
    cpd_pneu = model_top.get_cpds("pneu")
    cpd_inf = model_top.get_cpds("inf")
    cpd_season = model_top.get_cpds("season")

    # bottom part: train CPDs for dysp, cough and nasal 
    # NaN rows are automatically skipped over during fitting
    state_names = {"pneu": ["yes", "no"], "inf": ["yes", "no"],
                   "dysp": ["yes", "no"], "cough": ["yes", "no"], "nasal": ["yes", "no"]}
    model_bottom = BayesianNetwork([("pneu", "dysp"), ("pneu", "cough"),
                                    ("inf", "cough"), ("inf", "nasal")])
    model_bottom.fit(data[["pneu", "inf", "dysp", "cough", "nasal"]], estimator=BayesianEstimator, prior_type="K2", state_names=state_names)
    cpd_dysp = model_bottom.get_cpds("dysp")
    cpd_cough = model_bottom.get_cpds("cough")
    cpd_nasal = model_bottom.get_cpds("nasal")

    # construct the full model (top + bottom) and use the CPDs from the separate parts
    model_full = BayesianNetwork([("season", "pneu"), ("season", "inf"),
                                  ("pneu", "dysp"), ("pneu", "cough"),
                                  ("inf", "cough"), ("inf", "nasal")])
    model_full.add_cpds(cpd_pneu, cpd_inf, cpd_season, cpd_dysp, cpd_cough, cpd_nasal)

    return model_full

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

def fit_BN_plus_model(data): 
    """
    Fit Bayesian network to data WITH fever and pain variables
    Follow GT structure: season -> pneu, season -> inf, pneu -> dysp, pneu -> cough, inf -> cough, inf -> nasal
                         pneu -> fever, inf -> fever, pneu -> pain, inf -> pain

    the rows with NaNs for the symptoms should still contribute to training of the CPDs for season, pneu and inf. 
    but in the standard pgmpy fit method, the rows with empty values are simply skipped over 
    we should be able to use Expectation Maximization to deal with this, but this does not work in pgmpy
    since the missing values are always leaf nodes, we can solve the problem by splitting 
    the BN in a top and a bottom part, and training them separately with the full dataset.
    """ 

    # top part: train CPDs for pneu, inf and season
    state_names = {"pneu": ["yes", "no"], "inf": ["yes", "no"], "season": ["warm", "cold"]}
    model_top = BayesianNetwork([("season", "pneu"), ("season", "inf")])
    model_top.fit(data[["pneu", "inf", "season"]], estimator=BayesianEstimator, prior_type="K2", state_names=state_names)
    cpd_pneu = model_top.get_cpds("pneu")
    cpd_inf = model_top.get_cpds("inf")
    cpd_season = model_top.get_cpds("season")

    # bottom part: train CPDs for dysp, cough, nasal, fever and pain
    # NaN rows are automatically skipped over during fitting
    state_names = {"pneu": ["yes", "no"], "inf": ["yes", "no"],
                   "dysp": ["yes", "no"], "cough": ["yes", "no"], "nasal": ["yes", "no"],
                   "pain": ["yes", "no"], "fever": ["high", "low", "none"]}
    model_bottom = BayesianNetwork([("pneu", "dysp"), ("pneu", "cough"),
                                    ("inf", "cough"), ("inf", "nasal"),
                                    ("pneu", "fever"), ("inf", "fever"), 
                                    ("pneu", "pain"), ("inf", "pain")])
    model_bottom.fit(data[["pneu", "inf", "dysp", "cough", "nasal", "fever", "pain"]], estimator=BayesianEstimator, prior_type="K2", state_names=state_names)
    cpd_dysp = model_bottom.get_cpds("dysp")
    cpd_cough = model_bottom.get_cpds("cough")
    cpd_nasal = model_bottom.get_cpds("nasal")
    cpd_fever = model_bottom.get_cpds("fever")
    cpd_pain = model_bottom.get_cpds("pain")

    # construct the full model (top + bottom) and use the CPDs from the separate parts
    model_full = BayesianNetwork([("season", "pneu"), ("season", "inf"),
                                  ("pneu", "dysp"), ("pneu", "cough"),
                                  ("inf", "cough"), ("inf", "nasal"), 
                                  ("pneu", "fever"), ("inf", "fever"), 
                                  ("pneu", "pain"), ("inf", "pain")])
    model_full.add_cpds(cpd_pneu, cpd_inf, cpd_season, cpd_dysp, cpd_cough, cpd_nasal, cpd_fever, cpd_pain)

    return model_full