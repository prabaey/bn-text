import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pgmpy.inference.ExactInference import VariableElimination
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

def calc_prob_CPT(df, combo): 
    """
    evaluate joint probability distribution
    P(season, pneu, inf, dysp, cough, nasal) = P(season)*P(pneu|season)*P(inf|season)*P(dysp|pneu)*P(cough|pneu,inf)*P(nasal|inf)

    df: dataframes containing conditional probabilities
    combo: the variable instantiation to evaluate

    returns: probability P(season, pneu, inf, dysp, cough, nasal)
    """
    val = {"season": combo[0], "pneu": combo[1], "inf": combo[2], "dysp": combo[3], "cough": combo[4], "nasal":combo[5]} 
    prob = df["season"].iloc[val["season"], 0] 
    prob *= df["pneu"].iloc[val["pneu"], val["season"]]
    prob *= df["inf"].iloc[val["inf"], val["season"]]
    prob *= df["dysp"].iloc[val["dysp"], val["pneu"]]
    prob *= df["nasal"].iloc[val["nasal"], val["inf"]]
    idx_cough = 2*(val["pneu"]) + val["inf"]
    prob *= df["cough"].iloc[val["cough"], idx_cough]
    return prob

def CPD_params(model, model_type, empty_text_emb=None): 
    """
    compares ground truth probability tables with learned probability tables. 
    calculates the KL divergence between the ground truth joint probability distribution and the learned joint probability distribution. 
    
    model: Bayesian network with learned conditional probabilities
    model_type: type of model (BN, ff, discr, gen, discr_abl)

    returns
        df_GT: conditional probability tables in ground truth Bayesian network
        df_est: estimated conditional probability tables from model
        KL: KL divergence between the ground truth joint distribution and the estimated joint distribution
    """

    # ground truth CPD parameters
    df_GT = {}
    df_GT["season"] = pd.DataFrame({"": [0.4, 0.6]}, index=["season = cold", "season = warm"])
    df_GT["pneu"] = pd.DataFrame({"season = warm": [0.005, 0.995], 
                            "season = cold": [0.015, 0.985]}, index=["pneu = yes", "pneu = no"])
    df_GT["inf"] = pd.DataFrame({"season = warm": [0.05, 0.95], 
                           "season = cold": [0.5, 0.5]}, index=["inf = yes", "inf = no"])
    df_GT["dysp"] = pd.DataFrame({"pneu = yes": [0.3, 0.7],
                            "pneu = no": [0.15, 0.85]}, index=["dysp = yes", "dysp = no"])
    df_GT["nasal"] = pd.DataFrame({"inf = yes": [0.7, 0.3], 
                             "inf = no": [0.2, 0.8]}, index=["nasal = yes", "nasal = no"])
    df_GT["cough"] = pd.DataFrame({"pneu = yes, inf = yes": [0.9, 0.1], 
                             "pneu = yes, inf = no": [0.9, 0.1], 
                             "pneu = no, inf = yes": [0.8, 0.2], 
                             "pneu = no, inf = no": [0.05, 0.95]}, index=["cough = yes", "cough = no"])

    # CPD parameters estimated from BN model
    if model_type == "BN":
        inf = VariableElimination(model)

        seas_prob = inf.query(["season"], evidence={}, show_progress=False).get_value(**{"season":"cold"})

        pneu_prob = [0, 0]
        pneu_prob[0] = inf.query(["pneu"], evidence={"season": "warm"}, show_progress=False).get_value(**{"pneu":"yes"})
        pneu_prob[1] = inf.query(["pneu"], evidence={"season": "cold"}, show_progress=False).get_value(**{"pneu":"yes"})

        inf_prob = [0, 0]
        inf_prob[0] = inf.query(["inf"], evidence={"season": "warm"}, show_progress=False).get_value(**{"inf":"yes"})
        inf_prob[1] = inf.query(["inf"], evidence={"season": "cold"}, show_progress=False).get_value(**{"inf":"yes"})

        dysp_prob = [0, 0]
        dysp_prob[0] = inf.query(["dysp"], evidence={"pneu": "no"}, show_progress=False).get_value(**{"dysp":"yes"})
        dysp_prob[1] = inf.query(["dysp"], evidence={"pneu": "yes"}, show_progress=False).get_value(**{"dysp":"yes"})

        nasal_prob = [0, 0]
        nasal_prob[0] = inf.query(["nasal"], evidence={"inf": "no"}, show_progress=False).get_value(**{"nasal":"yes"})
        nasal_prob[1] = inf.query(["nasal"], evidence={"inf": "yes"}, show_progress=False).get_value(**{"nasal":"yes"})

        cough_prob = [0, 0, 0, 0]
        cough_prob[0] = inf.query(["cough"], evidence={"pneu": "no", "inf": "no"}, show_progress=False).get_value(**{"cough":"yes"})
        cough_prob[1] = inf.query(["cough"], evidence={"pneu": "no", "inf": "yes"}, show_progress=False).get_value(**{"cough":"yes"})
        cough_prob[2] = inf.query(["cough"], evidence={"pneu": "yes", "inf": "no"}, show_progress=False).get_value(**{"cough":"yes"})
        cough_prob[3] = inf.query(["cough"], evidence={"pneu": "yes", "inf": "yes"}, show_progress=False).get_value(**{"cough":"yes"})

    # CPD parameters estimated by generative model
    if model_type == "gen": 
        seas_prob = model.dist["season"].probs.item()
        pneu_prob = [model.dist["pneu"].probs[i].item() for i in range(2)]
        inf_prob = [model.dist["inf"].probs[i].item() for i in range(2)]
        dysp_prob = [model.dist["dysp"].probs[i].item() for i in range(2)]
        nasal_prob = [model.dist["nasal"].probs[i].item() for i in range(2)]
        cough_prob = [model.dist["cough"].probs.flatten()[i].item() for i in range(4)]

    # CPD parameters estimated by discriminative model
    if model_type == "discr": 
        seas_prob = model.dist_season.probs.item()
        pneu_prob = [torch.sigmoid(model.classifiers["pneu"][i](empty_text_emb)).item() for i in range(2)]
        inf_prob = [torch.sigmoid(model.classifiers["inf"][i](empty_text_emb)).item() for i in range(2)]
        dysp_prob = [torch.sigmoid(model.classifiers["dysp"][i](empty_text_emb)).item() for i in range(2)]
        nasal_prob = [torch.sigmoid(model.classifiers["nasal"][i](empty_text_emb)).item() for i in range(2)]
        cough_prob = [torch.sigmoid(model.classifiers["cough"][i](empty_text_emb)).item() for i in range(4)]

    # CPD parameters estimated by ablated discriminative model
    if model_type == "discr_abl": 
        seas_prob = model.dist["season"].probs.item()
        pneu_prob = [model.dist["pneu"].probs[i].item() for i in range(2)]
        inf_prob = [model.dist["inf"].probs[i].item() for i in range(2)]
        dysp_prob = [torch.sigmoid(model.classifiers["dysp"][i](empty_text_emb)).item() for i in range(2)]
        nasal_prob = [torch.sigmoid(model.classifiers["nasal"][i](empty_text_emb)).item() for i in range(2)]
        cough_prob = [torch.sigmoid(model.classifiers["cough"][i](empty_text_emb)).item() for i in range(4)]
 
    # build up conditional probability tables
    df_est = {}
    df_est["season"] = pd.DataFrame({"": [seas_prob, 1-seas_prob]}, index=["season = cold", "season = warm"])
    df_est["pneu"] = pd.DataFrame({"season = warm": [pneu_prob[0], 1-pneu_prob[0]], 
                                   "season = cold": [pneu_prob[1], 1-pneu_prob[1]]}, index=["pneu = yes", "pneu = no"])
    df_est["inf"] = pd.DataFrame({"season = warm": [inf_prob[0], 1-inf_prob[0]], 
                                  "season = cold": [inf_prob[1], 1-inf_prob[1]]}, index=["inf = yes", "inf = no"])
    df_est["dysp"] = pd.DataFrame({"pneu = yes": [dysp_prob[1], 1-dysp_prob[1]],
                                   "pneu = no": [dysp_prob[0], 1-dysp_prob[0]]}, index=["dysp = yes", "dysp = no"])
    df_est["nasal"] = pd.DataFrame({"inf = yes": [nasal_prob[1], 1-nasal_prob[1]], 
                                    "inf = no": [nasal_prob[0], 1-nasal_prob[0]]}, index=["nasal = yes", "nasal = no"])
    df_est["cough"] = pd.DataFrame({"pneu = yes, inf = yes": [cough_prob[3], 1-cough_prob[3]], 
                                    "pneu = yes, inf = no": [cough_prob[2], 1-cough_prob[2]], 
                                    "pneu = no, inf = yes": [cough_prob[1], 1-cough_prob[1]], 
                                    "pneu = no, inf = no": [cough_prob[0], 1-cough_prob[0]]}, index=["cough = yes", "cough = no"])
    
    # calculate KL divergence based on joint probability distribution for both CPD configurations
    val_combos = list(itertools.product([0, 1], repeat=6))
    KL = 0
    for combo in val_combos: # sum over all possible x
        prob_GT = calc_prob_CPT(df_GT, combo) # P(x)
        prob_est = calc_prob_CPT(df_est, combo) # Q(x)
        if prob_est == 0: 
            prob_est += 1e-8 # avoid division by zero error
        KL += prob_GT*np.log(prob_GT/prob_est) # P(x)*log(P(x)/Q(x))

    return df_GT, df_est, KL

def performance_metrics(pred_df, diag, model_type, plot=False): 
    """
    Calculates ROC and average precision (APS), and plots the curves if requested. 

    pred_df: dataframe containing predictions for the diagnosis ("pred_pneu" or "pred_inf")
    diag: name of the diagnosis for which metrics are calculated 
    model_type: type of model (BN, BN_plus, ff, discr, gen, discr_abl)
    plot: whether to plot ROC and precision-recall curves

    returns: 
        roc_auc: area under ROC curve
        aps: Average Precision (area under precision-recall curve)
    """

    if model_type == "BN" or model_type == "BN_plus":
        fpr, tpr, _ = roc_curve(pred_df[diag], pred_df[f"pred_{diag}"], pos_label="yes")
    else: 
        fpr, tpr, _ = roc_curve(pred_df[diag], pred_df[f"pred_{diag}"])
    roc_auc = auc(fpr, tpr)

    if plot: 
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, marker="o", label=f"ROC curve, AUC={roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle='--', marker="o", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve for {diag}")
        plt.legend()
        plt.show()

    if model_type == "BN" or model_type == "BN_plus": 
        aps = average_precision_score(pred_df[diag], pred_df[f"pred_{diag}"], pos_label="yes")
        precision, recall, _ = precision_recall_curve(pred_df[diag], pred_df[f"pred_{diag}"], pos_label="yes")
    else: 
        aps = average_precision_score(pred_df[diag], pred_df[f"pred_{diag}"])
        precision, recall, _ = precision_recall_curve(pred_df[diag], pred_df[f"pred_{diag}"])

    if plot: 
        plt.figure(figsize=(8,8))
        plt.plot(recall, precision, marker="o", label=f"PR curve, AP={aps:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall curve for {diag}")
        plt.legend()
        plt.show()

    return roc_auc, aps


def plot_loss(train_loss_individual, test_loss_individual, epochs, train_loss, test_loss): 
    """
    plot losses across epochs

    train_loss_individual: dict of individual train losses across epochs (i.e. logP for each separate conditional distribution)
    test_loss_individual: dict of individual test losses across epochs (i.e. logP for each separate conditional distribution)
    epochs: number of epochs
    train_loss: total train loss across epochs
    test_loss: total test loss across epochs
    """

    # plot total train and test loss over epochs
    plt.plot(np.arange(epochs), train_loss, label="train")
    plt.plot(np.arange(epochs), test_loss, label="test")
    plt.legend()
    plt.title("train and test loss")
    plt.show()

    # plot losses over epochs for all individual variables
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

    key_it = list(train_loss_individual.keys())
    for i, ax in enumerate(axes.flat):
        key = key_it[i]
        ax.plot(np.arange(epochs), train_loss_individual[key], label=f"{key} train")
        ax.plot(np.arange(epochs), test_loss_individual[key], label=f"{key} test")
        ax.legend()
        
    plt.show()


def infer_query(model, query_var, query_val, evidence, all_vars, model_type=None): 
    """ 
    Calculate logP(query_var=query_val | evidence) through Bayesian inference
    logP(query_var=query_val|evidence) = logP(query_var=query_val,evidence) - logP(query_var=query_val)
                                       = NOMINATOR - DENOMINATOR
    both nominator and denominator are calculated by marginalizing unobserved variables (=variables not part of evidence) 
    out of the joint probability distribution (=summing over all possible values for those variables)
    This mimics the behaviour of exact inference in Bayesian networks, except we use the outputs of an arbitrary model 

    model: trained model, can contain text node (generative text model or discriminative text model)
    query_var: query variable (pneu/inf)
    query_val: value of query variable (yes/no)
    evidence: dict of named evidence variables and their corresponding values, e.g. {"season": 1"dysp": 1, "cough": 0, "nasal": 1}
    all_vars: set of BN node names, excluding text ({"season", "pneu", "inf", "dysp", "cough", "nasal"})
    model_type: type of model ("gen", "discr" or "discr_abl")
    """
    # get logP(query_var = query_val | evidence) according to model, using bayesian network structure

    ev_keys = set(evidence.keys())
    marg_set_nom = all_vars - ev_keys.union(set([query_var])) # set of vars to marginalize over in nominator

    # NOMINATOR

    # get all possible value combinations for unobserved variables
    val_combinations = list(itertools.product([torch.tensor([0.0]), torch.tensor([1.0])], repeat=len(marg_set_nom)))
    val_combinations = torch.tensor(np.array(val_combinations).T[0])

    # build up model input with evidence and query var
    x = {}
    for key, val in evidence.items(): 
        if key == "text": 
            x[key] = val.repeat(val_combinations.shape[1], 1)
        else: 
            x[key] = val.repeat(val_combinations.shape[1])
    x[query_var] = query_val.repeat(val_combinations.shape[1])
    
    # add values for unobserved variables
    for i, key in enumerate(marg_set_nom): 
        x[key] = val_combinations[i]

    # get log-prob of full variable instantiation
    # P(query var = query val, evidence, other vars = val_combinations | text), for all val_combinations
    with torch.no_grad(): 
        if model_type == "discr" or model_type == "discr_abl":
            res = model(x)
            logp_nom = sum(res.values()) 
        elif model_type == "gen" and ("text" in ev_keys): 
            logp_nom, _ = model.log_lik_total(x)
        elif model_type == "gen" and ("text" not in ev_keys): # evaluate generative model without text 
            logp_nom, _, _ = model(x)

    # DENOMINATOR

    x[query_var] = (1-query_val).repeat(val_combinations.shape[1]) # get log prob for compliment of query value

    # get log-prob of full variable instantiation
    # P(query var = 1-query val, evidence, other vars = val_combinations | text), for all val_combinations
    with torch.no_grad(): 
        if model_type == "discr" or model_type == "discr_abl":
            res = model(x)
            logp_compl = sum(res.values()) 
        elif model_type == "gen" and ("text" in ev_keys): 
            logp_compl, _ = model.log_lik_total(x)
        elif model_type == "gen" and ("text" not in ev_keys): # evaluate generative model without text 
            logp_compl, _, _ = model(x)
    logp_denom = torch.cat([logp_nom, logp_compl]) # denominator needs to marginalize over query var as well

    # NOMINATOR / DENOMINATOR

    logp_nom = torch.logsumexp(logp_nom, dim=0, keepdim=False) # sum all probabilities in nominator
    logp_denom = torch.logsumexp(logp_denom, dim=0, keepdim=False) # sum all probabilities in denominator

    return logp_nom - logp_denom # cond prob is nom/denom
    

def predict_diagnoses(model, test_set, bs, model_type=None, excl_text=False, excl_sympt=False, empty_text_emb=None):
    """ 
    build prediction dataframe for diagnoses (pneu, inf) based on predictions made by trained model for set of test samples

    model: trained model (generative or discriminative)
    test_set: dataframe with test cases
    bs: batch size to use when looping over test cases
    model_type: type of model ("gen", "discr" or "discr_abl")
    excl_text: if True, calculate P(diag=yes|background,symptoms)
    excl_sympt: if True, calculate P(diag=yes|background,text)
        if excl_text and excl_sympt are both False, we calculate P(diag=yes|background,symptoms,text)
    empty_text_emb: embedding of empty text "" to use when excl_text=True

    returns: test_set dataframe, extended with prediction for diagnosis (pred_pneu/pred_inf)
    """

    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True)
    res_df = pd.DataFrame(columns=list(test_set.__getitem__(0).keys()))
    res_df = res_df.rename({"text": "emb"}, axis=1)

    all_vars = set(["pneu", "inf", "season", "dysp", "nasal", "cough"])

    model.eval() # put model in evaluation mode

    if model_type == "gen" and bs != 1: 
        bs = 1 # batch size must be 1 when using infer_query, which is not parallelizable
        print("Batch size must be 1 when infer_query function is called.")
    if model_type == "discr" and not excl_sympt and bs != 1: 
        bs = 1 # batch size must be 1 when using infer_query, which is not parallelizable
        print("Batch size must be 1 when infer_query function is called.")

    for x in test_loader: 
        with torch.no_grad(): 

            if model_type == "discr":
                if excl_text: # use empty text embedding at input of classifiers
                    x["text"] = empty_text_emb.unsqueeze(0)
                if excl_sympt: # just get output of classifiers
                    res_pneu = model.log_lik_class(x, "pneu", ["season"], pred_prob=True) # get P(pneu = yes | background, text)
                    res_inf = model.log_lik_class(x, "inf", ["season"], pred_prob=True) # get P(inf = yes | background, text)
                else: # apply bayesian inference, 
                    evidence = {key:val.cpu() for key, val in x.items() if key not in ["pneu", "inf"]}
                    res_pneu = infer_query(model, "pneu", torch.tensor([1.]), evidence, all_vars, model_type=model_type).exp() # P(pneu = yes | background, symptoms, text)
                    res_inf = infer_query(model, "inf", torch.tensor([1.]), evidence, all_vars, model_type=model_type).exp() # P(inf = yes | background, symptoms, text)

            if model_type == "discr_abl":
                if excl_text: # use empty text embedding at input of classifiers
                    x["text"] = empty_text_emb.unsqueeze(0)
                if excl_sympt: # in ablated model, diagnoses are independent of text if no symptoms are provided
                    x_pos = x.copy()
                    x_pos["pneu"] = torch.tensor([1.]*x["pneu"].shape[0])
                    x_pos["inf"] = torch.tensor([1.]*x["inf"].shape[0])
                    res_pneu = model.log_p_cond(x_pos, "pneu", ["season"]).exp() # P(pneu = yes | background, text) = P(pneu = yes | background)
                    res_inf = model.log_p_cond(x_pos, "inf", ["season"]).exp() # P(inf = yes | background, text) = P(inf = yes | background)
                else: 
                    evidence = {key:val.cpu() for key, val in x.items() if key not in ["pneu", "inf"]}
                    res_pneu = infer_query(model, "pneu", torch.tensor([1.]), evidence, all_vars, model_type=model_type).exp() # P(pneu = yes | background, symptoms, text)
                    res_inf = infer_query(model, "inf", torch.tensor([1.]), evidence, all_vars, model_type=model_type).exp() # P(inf = yes | background, symptoms, text)

            if model_type == "gen":
                if excl_sympt: 
                    res_pneu = infer_query(model, "pneu", torch.tensor([1.]), {"season": x["season"], "text": x["text"]}, all_vars, model_type="gen").exp() # get P(pneu = yes | background, text)
                    res_inf = infer_query(model, "inf", torch.tensor([1.]), {"season": x["season"], "text": x["text"]}, all_vars, model_type="gen").exp() # get P(inf = yes | background, text)
                elif excl_text: 
                    res_pneu = infer_query(model, "pneu", torch.tensor([1.]), {"season": x["season"], "dysp": x["dysp"], "cough": x["cough"], "nasal": x["nasal"]}, all_vars, model_type="gen").exp() # get P(pneu = yes | background, symptoms)
                    res_inf = infer_query(model, "inf", torch.tensor([1.]), {"season": x["season"], "dysp": x["dysp"], "cough": x["cough"], "nasal": x["nasal"]}, all_vars, model_type="gen").exp() # get P(inf = yes | background, symptoms)
                else: 
                    res_pneu = infer_query(model, "pneu", torch.tensor([1.]), {"season": x["season"], "dysp": x["dysp"], "cough": x["cough"], "nasal": x["nasal"], "text": x["text"]}, all_vars, model_type="gen").exp() # get P(pneu = yes | background, symptoms, text)
                    res_inf = infer_query(model, "inf", torch.tensor([1.]), {"season": x["season"], "dysp": x["dysp"], "cough": x["cough"], "nasal": x["nasal"], "text": x["text"]}, all_vars, model_type="gen").exp() # get P(inf = yes | background, symptoms, text)

        batch_df = pd.DataFrame({key:val.cpu().numpy() for key, val in x.items() if key != "text"})
        batch_df["pred_pneu"] = res_pneu.cpu().numpy()
        batch_df["pred_inf"] = res_inf.cpu().numpy()
        batch_df["emb"] = list(x["text"].cpu())
        res_df = pd.concat([res_df, batch_df], ignore_index=True)

    return res_df