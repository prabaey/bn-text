import torch
from torch import nn
from torch.distributions import Bernoulli, MultivariateNormal

class TextEmbClassifier(nn.Module):
    """
    Classifier that takes text embeddings as an input, and outputs a logit that can be transformed into a diagnosis probability

    n_emb: dimension of text embedding input
    hidden_dim: list of dimensions of hidden layers. if empty, no transformation is applied. if len>0, final dimension should be 1
    dropout_prob: dropout probability to be applied before every hidden layer.
    seed: initialization seed
    """
     
    def __init__(self, n_emb, hidden_dim, dropout_prob, seed, apply_sigmoid=True): 

        super(TextEmbClassifier, self).__init__()

        torch.manual_seed(seed)

        self.n_emb = n_emb # embedding size of text 
        self.hidden_dim = hidden_dim # hidden dimension, if len == 0 then no transformation is applied
                                     # if len != 0, final dimension should be 1 to allow for classification
        self.apply_sigmoid = apply_sigmoid # whether to apply sigmoid at the end of forward pass

        # initialize parameters
        if len(hidden_dim) == 0: 
            self.linears = nn.ModuleList([])
        else:  
            self.linears = nn.ModuleList([nn.Linear(self.n_emb, hidden_dim[0])])
            self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob)])
            prev_dim = hidden_dim[0]
            for dim in hidden_dim[1:]: 
                layer = nn.Linear(prev_dim, dim)
                dropout = nn.Dropout(p=dropout_prob)
                self.linears.append(layer)
                self.dropouts.append(dropout)
                prev_dim = dim
            self.hidden_activation = nn.ReLU() # ReLU is used as activation after every hidden layer

    def forward(self, emb): 
        """
        forward function. transforms embedding of dim n_emb to output of size 1 by applying linear layers
        """

        if len(self.hidden_dim) == 0: 
            return emb
        else: 
            out = emb
            for i, layer in enumerate(self.linears[:-1]): 
                out = self.dropouts[i](out)
                out = layer(out)
                out = self.hidden_activation(out) # ReLU for activation between hidden layers
            out = self.dropouts[-1](out) # if only one layer, dropout should be applied to inputs
            out = self.linears[-1](out)

            if self.apply_sigmoid:
                return out.sigmoid() # sigmoid at the end
            else: 
                return out # no sigmoid at the end
        
    
class GenerativeModel(nn.Module):
    """
    Generative model, where conditional probabilities for season, pneu, inf, dysp, cough and nasal are parameterized
    using a Bernoulli distribution, with one learnable parameter per parent configuration. 
    Each conditional text distribution (one per possible configuration of parents pneu, inf, dysp, cough and nasal) 
    is parameterized by a Gaussian, with mean and covariance pre-fitted on the text embeddings in the training set
    """
    
    def __init__(self, seed, device): 
        super(GenerativeModel, self).__init__()

        torch.manual_seed(seed)
        
        # initialize Bernoulli parameters for conditional probability tables
        self.params = nn.ParameterDict({
            "season": nn.Parameter(torch.rand(1)),  # P(season)
            "pneu": nn.Parameter(torch.rand(2)),    # [P(pneu=1|season=0), P(pneu=1|season=1)]
            "inf": nn.Parameter(torch.rand(2)),     # [P(inf=1|season=0), P(inf=1|season=1)]
            "dysp": nn.Parameter(torch.rand(2)),    # [P(dysp=1|pneu=0), P(dysp=1|pneu=1)]
            "cough": nn.Parameter(torch.rand(2,2)), # [[P(cough=1|pneu=0,inf=0), P(cough=1|pneu=0,inf=1)], [P(cough=1|pneu=1,inf=0), P(cough=1|pneu=1,inf=1)]]
            "nasal": nn.Parameter(torch.rand(2))    # [P(nasal=1|inf=0), P(nasal=1|inf=1)]
        })

        self.p_text =  [None, None, # P(Pneu=0,Inf=0,Dysp=0,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=0,Dysp=0,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=0,Dysp=1,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=0,Dysp=1,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=1,Dysp=0,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=1,Dysp=0,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=1,Dysp=1,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=0,Inf=1,Dysp=1,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=0,Dysp=0,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=0,Dysp=0,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=0,Dysp=1,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=0,Dysp=1,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=1,Dysp=0,Cough=0,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=1,Dysp=0,Cough=1,Nasal=0/1)
                        None, None, # P(Pneu=1,Inf=1,Dysp=1,Cough=0,Nasal=0/1)
                        None, None] # P(Pneu=1,Inf=1,Dysp=1,Cough=1,Nasal=0/1)

        self.device = device

        # create conditional probability distributions from Bernoulli parameters
        self.dist = {}
        self.update_dist()
    
    def add_cond_text_distr(self, idx, embs,): 
        """ 
        Add Gaussian to the model, parameterizing conditional text distribution

        idx: index of conditional Gaussian in self.p_text, see corresponding parent configurations in init
        embs: tensor of embeddings that are used to fit this Gaussian. these should be selected from the training dataset according 
              to the parent condition corresponding to the conditional text distribution you want to parameterize
              e.g. if idx = 0, then embs contains all non-empty text embeddings from training set where {Pneu=0,Inf=0,Dysp=0,Cough=0,Nasal=0}
        """

        if len(embs) == 0:
            mu = torch.zeros(self.n_emb) # if there are not enough samples to compute a covariance matrix, just use 0 as mean
        else:
            mu = torch.mean(embs, axis=0)
        cov = torch.cov(embs.t()) # from docs of cov function: rows are variables, columns are observations -> need to transpose emb
        if torch.isnan(cov).any(): # if there are not enough samples to compute a covariance matrix, just use the identity matrix
            cov = torch.eye(self.n_emb)
        sigma = (1-self.alpha)*cov + self.alpha*torch.eye(cov.shape[0]) # regularization, see QDA sklearn

        self.p_text[idx] = MultivariateNormal(mu, sigma) # n_emb-dimensional Gaussian

    def add_gaussians(self, train_df, alpha, n_emb):
        """
        Go through all possible joint configurations of parents (pneu,inf,dysp,cough,nasal), select embeddings in train set
        that fit this condition, and use method add_cond_text_distr with appropriate index to fit the conditional Gaussian and 
        store it in the model. 

        train_df: train set (dataframe)
        alpha: regularization parameter (hyperparameter). allows tuning the contribution of the individual variances of the text representation dimensions
        n_emb: text embedding dimension
        """
        self.alpha = alpha
        self.n_emb = n_emb

        idx = 0
        for pneu_val in ["no", "yes"]:
            for inf_val in ["no", "yes"]: 
                for dysp_val in ["no", "yes"]:
                    for cough_val in ["no", "yes"]:
                        for nasal_val in ["no", "yes"]:
                            df_pneu = train_df[train_df["pneu"] == pneu_val]
                            df_inf = df_pneu[df_pneu["inf"] == inf_val]
                            df_dysp = df_inf[df_inf["dysp"] == dysp_val]
                            df_cough = df_dysp[df_dysp["cough"] == cough_val]
                            df_nasal = df_cough[df_cough["nasal"] == nasal_val]
                            df_subset = df_nasal
                            embs = torch.tensor(list(df_subset["BioLORD emb"]))
                            self.add_cond_text_distr(idx, embs)
                            idx += 1


    def log_lik_text(self, x):
        """ 
        Calculate logP(text|pneu,inf,dysp,cough,nasal) for values in sample x
        """

        logp_text = [gauss.log_prob(x["text"]).unsqueeze(1) for gauss in self.p_text]
        logp_text = torch.cat(logp_text, dim=1) # shape (bs, 32)

        # create condition to select correct gaussian according to diagnosis and symptom values
        cond1 = x["pneu"]
        cond2 = x["inf"]
        cond3 = x["dysp"]
        cond4 = x["cough"]
        cond5 = x["nasal"]
        cond = 16*cond1 + 8*cond2 + 4*cond3 + 2*cond4 + cond5 # combined condition, shape (bs,)

        logp_text = torch.gather(logp_text, 1, cond[:,None].long()).squeeze(1)

        return logp_text # shape (bs,)
    
    def log_lik_total(self, x): 
        """ 
        Calculate logP(season,pneu,inf,dysp,cough,nasal,text) for values in sample x
        = logP(season)+logP(pneu|season)+logP(inf|season)+logP(dysp|pneu)+logP(cough|pneu,inf)+logP(nasal|inf)
          + logP(text|pneu,inf,dysp,cough,nasal)

        returns
            sum(log_p.values()): total logP(season,pneu,inf,dysp,cough,nasal,text)
            log_p: dictionary of individual conditional probabilities
        """

        log_p = {}
        log_p["season"] = self.dist["season"].log_prob(x["season"]) # logP(Season)
        log_p["pneu"] = self.log_p_cond(x, "pneu", ["season"]) # logP(Pneu | Season)
        log_p["inf"] = self.log_p_cond(x, "inf", ["season"])  # logP(Inf | Season)
        log_p["dysp"] = self.log_p_cond(x, "dysp", ["pneu"]) # logP(Dysp | Pneu)
        log_p["cough"] = self.log_p_cond(x, "cough", ["pneu", "inf"]) # logP(Cough | Pneu, Inf)
        log_p["nasal"] = self.log_p_cond(x, "nasal", ["inf"]) # logP(Nasal | Inf)

        log_p["text"] = self.log_lik_text(x) # logP(Text|Pneu,Inf,Dysp,Cough,Nasal)

        return sum(log_p.values()), log_p

    def update_dist(self): 
        """ 
        Update Bernoulli distributions based on conditional parameters
        """
        
        # update distributions
        self.dist["season"] = Bernoulli(probs=torch.sigmoid(self.params["season"]))
        self.dist["pneu"] = Bernoulli(probs=torch.sigmoid(self.params["pneu"]))
        self.dist["inf"] = Bernoulli(probs=torch.sigmoid(self.params["inf"]))
        self.dist["dysp"] = Bernoulli(probs=torch.sigmoid(self.params["dysp"]))
        self.dist["cough"] = Bernoulli(probs=torch.sigmoid(self.params["cough"]))
        self.dist["nasal"] = Bernoulli(probs=torch.sigmoid(self.params["nasal"]))
    
    def log_p_cond(self, x, child, parents): 
        """ 
        Calculates P(child|parents) where parents can have arbitrary length, assuming each parent can take on 2 values 
        x: sample containing values for all variables
        child: name of child ("pneu", "inf", "dysp", "cough" or "nasal")
        parents: list of names of parents (e.g., for "cough" this is ["pneu", "inf"])
        returns: P(child|parents)
        """

        n = len(parents)

        x_child = x[child] 
        x_child = x_child[(...,) + (None,)*n]       # equivalent to x_child[:, None, None, ..., None] with number of Nones equal to n
        prob = self.dist[child].log_prob(x_child)   # get prob P(Child=c|Parents) for all possible parent values
                                                    # shape (bs, (2,)*n), e.g. (bs, 2, 2, 2) for 3 parents
        for i in range(n): 
            parent = parents[i]
            x_parent = x[parent]
            x_parent = x_parent[(...,) + (None,)*(n-i)].expand(-1, 1, *(2,)*(n-i-1)) # shape (bs, 1, (2,)*(n-i-1))
            prob = torch.gather(prob, 1, x_parent.long()).squeeze(1) # select prob P(Child=c|Parent=p,Remaining parents)
                                                                     # shape (bs, (2,)*(n-i)) -> n-i parents remain         
        return prob
    
    def fully_obs_CPT(self, x): 
        """ 
        Calculate logP(season,pneu,inf,dysp,cough,nasal) for values in fully observed sample x (i.e. no nan values for symptoms)
        = logP(season)+logP(pneu|season)+logP(inf|season)+logP(dysp|pneu)+logP(cough|pneu,inf)+logP(nasal|inf)
        Is used to train CPT portion of the BN (so conditional probability tables for all variables excluding text)

        x: sample containing observed values for all variables
        returns
            sum(log_p.values()): total logP(season,pneu,inf,dysp,cough,nasal)
            log_p: dictionary of individual conditional probabilities
        """
    
        log_p = {}

        # CPDs
        log_p["season"] = self.dist["season"].log_prob(x["season"]) # logP(Season)
        log_p["pneu"] = self.log_p_cond(x, "pneu", ["season"]) # logP(Pneu | Season)
        log_p["inf"] = self.log_p_cond(x, "inf", ["season"])  # logP(Inf | Season)
        log_p["dysp"] = self.log_p_cond(x, "dysp", ["pneu"]) # logP(Dysp | Pneu)
        log_p["cough"] = self.log_p_cond(x, "cough", ["pneu", "inf"]) # logP(Cough | Pneu, Inf)
        log_p["nasal"] = self.log_p_cond(x, "nasal", ["inf"]) # logP(Nasal | Inf)

        # logP(Season, Pneu, Inf, Dysp, Cough, Nasal) 
        # = logP(Season) + logP(Pneu | Season) + logP(Inf | Season)
        # + logP(Dysp | Pneu) + logP(Cough | Pneu, Inf) + logP(Nasal | Inf)
        return sum(log_p.values()), log_p

    def part_obs_CPT(self, x): 
        """ 
        Calculate logP(season,pneu,inf) for values in partially observed sample x (i.e. nan values for symptoms)
        = logP(season)+logP(pneu|season)+logP(inf|season)
        Is used to train CPT portion of the BN (so conditional probability tables for all variables excluding text)

        x: sample containing observed values for all non-symptom variables
        returns
            sum(log_p.values()): total logP(season,pneu,inf)
            log_p: dictionary of individual conditional probabilities
        """

        log_p = {}

        # CPDs
        log_p["season"] = self.dist["season"].log_prob(x["season"]) # logP(Season)
        log_p["pneu"] = self.log_p_cond(x, "pneu", ["season"]) # logP(Pneu | Season)
        log_p["inf"] = self.log_p_cond(x, "inf", ["season"])  # logP(Inf | Season)

        # logP(Season, Pneu, Inf) 
        # = logP(Season) + logP(Pneu | Season) + logP(Inf | Season)
        return sum(log_p.values()), log_p
        
    def forward(self, x):
        """ 
        Calculate logP of sample x under current parameterization of CPTs
        Combines logP(season,pneu,inf,dysp,cough,nasal) for fully observed portion of samples x 
            and logP(season,pneu,inf) for partially observed (= symptoms not observed) portion of samples x
        Is used to train CPT portion of the BN (so conditional probability tables for all variables excluding text)

        x: sample containing observed values for all variables, format {"season":tensor(), "pneu":tensor(), "inf":tensor(), "dysp": tensor(), "cough": tensor(), "nasal": tensor()} 
        returns
            log_p_fully_obs: total logP(season,pneu,inf,dysp,cough,nasal) for fully observed portion of samples x
            log_p_part_obs: total logP(season,pneu,inf) for partially observed (= symptoms not observed) portion of samples x
            log_p_fully_obs_per_var: individual conditional logP's for every var, for fully observed portion of samples x
        """

        # first update distributions with current parameters
        self.update_dist()

        # split batch into fully observed part and partially observed part 
        x_part_obs = {key:val[torch.isnan(x["dysp"])] for key, val in x.items()}
        x_fully_obs = {key:val[~torch.isnan(x["dysp"])] for key, val in x.items()}
    
        log_p_fully_obs, log_p_fully_obs_per_var = self.fully_obs_CPT(x_fully_obs)
        log_p_part_obs, _ = self.part_obs_CPT(x_part_obs)

        # return losses for all classifiers separately where possible (for fully obs), so they're easier to monitor
        return log_p_fully_obs, log_p_part_obs, log_p_fully_obs_per_var
    

class GenerativeModelAbl(GenerativeModel):
    """
    Ablated generative model, where conditional probabilities for season, pneu, inf, dysp, cough and nasal are parameterized
    using a Bernoulli distribution, with one learnable parameter per parent configuration. 
    Each conditional text distribution (one per possible configuration of parents dysp, cough and nasal) 
    is parameterized by a Gaussian, with mean and covariance pre-fitted on the text embeddings in the training set

    Inherits most methods from GenerativeModel, since only conditional text distribution is different
    """
    
    def __init__(self, seed, device): 
        super(GenerativeModel, self).__init__(seed, device)

        torch.manual_seed(seed)
        
        # initialize Bernoulli parameters for conditional probability tables
        self.params = nn.ParameterDict({
            "season": nn.Parameter(torch.rand(1)),  # P(season)
            "pneu": nn.Parameter(torch.rand(2)),    # [P(pneu=1|season=0), P(pneu=1|season=1)]
            "inf": nn.Parameter(torch.rand(2)),     # [P(inf=1|season=0), P(inf=1|season=1)]
            "dysp": nn.Parameter(torch.rand(2)),    # [P(dysp=1|pneu=0), P(dysp=1|pneu=1)]
            "cough": nn.Parameter(torch.rand(2,2)), # [[P(cough=1|pneu=0,inf=0), P(cough=1|pneu=0,inf=1)], [P(cough=1|pneu=1,inf=0), P(cough=1|pneu=1,inf=1)]]
            "nasal": nn.Parameter(torch.rand(2))    # [P(nasal=1|inf=0), P(nasal=1|inf=1)]
        })

        self.p_text =  [None, None, # P(Dysp=0,Cough=0,Nasal=0/1)
                        None, None, # P(Dysp=0,Cough=1,Nasal=0/1
                        None, None, # P(Dysp=1,Cough=0,Nasal=0/1)
                        None, None] # P(Dysp=1,Cough=1,Nasal=0/1)

        self.device = device

        # create conditional probability distributions from Bernoulli parameters
        self.dist = {}
        self.update_dist()

    def add_gaussians(self, train_df, alpha, n_emb):
        """
        Go through all possible joint configurations of parents (dysp,cough,nasal), select embeddings in train set
        that fit this condition, and use method add_cond_text_distr with appropriate index to fit the conditional Gaussian and 
        store it in the model. 

        train_df: train set (dataframe)
        alpha: regularization parameter (hyperparameter). allows tuning the contribution of the individual variances of the text representation dimensions
        n_emb: text embedding dimension
        """
        self.alpha = alpha
        self.n_emb = n_emb

        idx = 0
        for dysp_val in ["no", "yes"]:
            for cough_val in ["no", "yes"]:
                for nasal_val in ["no", "yes"]:
                    df_dysp = train_df[train_df["dysp"] == dysp_val]
                    df_cough = df_dysp[df_dysp["cough"] == cough_val]
                    df_nasal = df_cough[df_cough["nasal"] == nasal_val]
                    df_subset = df_nasal
                    embs = torch.tensor(list(df_subset["BioLORD emb"]))
                    self.add_cond_text_distr(idx, embs)
                    idx += 1

    def log_lik_text(self, x):
        """ 
        Calculate logP(text|dysp,cough,nasal) for values in sample x
        """

        logp_text = [gauss.log_prob(x["text"]).unsqueeze(1) for gauss in self.p_text]
        logp_text = torch.cat(logp_text, dim=1) # shape (bs, 8)

        # create condition to select correct gaussian according to diagnosis and symptom values
        cond1 = x["dysp"]
        cond2 = x["cough"]
        cond3 = x["nasal"]
        cond = 4*cond1 + 2*cond2 + cond3 # combined condition, shape (bs,)

        logp_text = torch.gather(logp_text, 1, cond[:,None].long()).squeeze(1)

        return logp_text # shape (bs,)
    

class DiscriminativeModel(nn.Module):
    """
    DiscriminativeModel: Bayesian network with discriminative text model
    The conditional probability distributions for nodes pneu, inf, dysp, cough and nasal are modeled by one text classifier
    per non-text parent configuration. The distribution P(Season) is modeled as a simple conditional probability table,
    parameterized by a Bernoulli parameter. 

    n_emb: text embedding dimension
    hidden_dim: dict {var_name: [...]} containing list of hidden dimensions to use in classifiers of "pneu", "inf", "dysp", "cough" and "nasal" nodes
    dropout: dict {var_name: ...} containing dropout probability to use in classifiers of "pneu", "inf", "dysp", "cough" and "nasal"
    seed: initialization seed 
    device: CPU/GPU
    """
    
    def __init__(self, n_emb, hidden_dim, dropout, seed, device): 
        super(DiscriminativeModel, self).__init__()

        torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # initialize parameters
        self.params = nn.ParameterDict({
            "season": nn.Parameter(torch.rand(1)), # P(season=cold)
        })
        self.classifiers = nn.ModuleDict({
            "pneu": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["pneu"], dropout["pneu"], seed, apply_sigmoid=False),     # P(pneu|season=warm,text)
                                   TextEmbClassifier(n_emb, hidden_dim["pneu"], dropout["pneu"], seed, apply_sigmoid=False)]) ,   # P(pneu|season=cold,text)
            "inf": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["inf"], dropout["inf"], seed, apply_sigmoid=False),      # P(inf|season=warm,text)
                                  TextEmbClassifier(n_emb, hidden_dim["inf"], dropout["inf"], seed, apply_sigmoid=False)]) ,    # P(inf|season=cold,text)
            "dysp": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["dysp"], dropout["dysp"], seed, apply_sigmoid=False),     # P(dysp|pneu=no,text)
                                   TextEmbClassifier(n_emb, hidden_dim["dysp"], dropout["dysp"], seed, apply_sigmoid=False)]),     # P(dysp|pneu=yes,text)
            "cough": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),    # P(cough|pneu=no,inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),   # P(cough|pneu=no,inf=yes,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),   # P(cough|pneu=yes,inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False)]),   # P(cough|pneu=yes,inf=yes,text)
            "nasal": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["nasal"], dropout["nasal"], seed, apply_sigmoid=False),    # P(nasal|inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["nasal"], dropout["nasal"], seed, apply_sigmoid=False)])    # P(nasal|inf=yes,text)
        })

        self.device = device


    def log_lik_class(self, x, child, parents, pred_prob=False):
        """ 
        calculate conditional probability logP(child|parents,text) of text classifier, given parent value and text embedding
        x: sample containing observed values for all variables (includes child and parent variable, and text embedding)
        child: name of child variable ("pneu", "inf", "dysp", "cough" or "nasal")
        parents: list of names of parent variables
        pred_prob: whether to output prediction probability of positive class instead of log-likelihood of sample
        returns: 
            if pred_prob == False: likelihood logP(child|parents,text), with child value as seen in x
            if pred_prob == True: prediction P(child=yes|parents,text), prob of child being positive
        """

        # get classifier outputs (=activation) for all possible parent values
        out = [cl(x["text"]) for cl in self.classifiers[child]]
        out = torch.cat(out, dim=1)
        
        if not pred_prob: 
            # get logprob by applying sigmoid and selecting log(P(child=1|...)) or log(1-P(child=1|...)) depending on value of child
            x_expand = torch.unsqueeze(x[child], dim=1).repeat(1, 2*len(parents))
            prob = -torch.nn.functional.binary_cross_entropy_with_logits(out, x_expand, reduction="none") # logP(child|parents,text)
        else: 
            prob = torch.sigmoid(out) # P(child=yes|parents,text)

        # create condition to select right outcome according to parent values
        if len(parents) == 1: # applicable to "pneu", "inf", "dysp" and "nasal", who all have one non-text parent node
            cond = x[parents[0]]
        elif len(parents) == 2: # applicable to "cough", who has two non-text parent nodes ("pneu" and "inf")
            cond1 = x[parents[0]]
            cond2 = x[parents[1]]
            cond = 2*cond1 + cond2 # combined condition
        else: # should not occur in this setup
            pass

        # select P(child|parent=p,text) according to parent conditions p
        prob = torch.gather(prob, 1, cond[:,None].long()).squeeze(1) # P(child|parent=p,text)

        return prob
        
    def forward(self, x):
        """ 
        Calculate logP of sample x under current parameterization of the CPT and text classifiers
        Combines logP(season,pneu,inf,dysp,cough,nasal|text) for fully observed portion of samples x 
            and logP(season,pneu,inf|text) for partially observed (= symptoms not observed) portion of samples x
        Is used to train all parameters in BN (so conditional probability table for season, and all 12 text classifiers)

        x: sample containing observed values for all variables, format {"season":tensor(), "pneu":tensor(), "inf":tensor(), "dysp": tensor(), "cough": tensor(), "nasal": tensor(), "text": tensor()} 
        returns: individual conditional log-probabilities for every variable, so they're easy to monitor separately during training
        """
        
        # split batch into fully observed part and partially observed part 
        x_part_obs = {key:val[torch.isnan(x["dysp"])] for key, val in x.items()}
        x_fully_obs = {key:val[~torch.isnan(x["dysp"])] for key, val in x.items()}
    
        # fully observed
        log_p_fully_obs = {}
        self.dist_season = Bernoulli(probs=torch.sigmoid(self.params["season"]))
        log_p_fully_obs["season"] = self.dist_season.log_prob(x_fully_obs["season"])          # logP(Season)
        log_p_fully_obs["pneu"] = self.log_lik_class(x_fully_obs, "pneu", ["season"])         # logP(Pneu | Season, Text)
        log_p_fully_obs["inf"] = self.log_lik_class(x_fully_obs, "inf", ["season"])           # logP(Inf | Season, Text)
        log_p_fully_obs["dysp"] = self.log_lik_class(x_fully_obs, "dysp", ["pneu"])           # logP(Dysp | Pneu, Text)
        log_p_fully_obs["cough"] = self.log_lik_class(x_fully_obs, "cough", ["pneu", "inf"])  # logP(Cough | Pneu, Inf, Text)
        log_p_fully_obs["nasal"] = self.log_lik_class(x_fully_obs, "nasal", ["inf"])          # logP(Nasal | Inf, Text)

        # partially observed
        log_p_part_obs = {}
        self.dist_season = Bernoulli(probs=torch.sigmoid(self.params["season"]))
        log_p_part_obs["season"] = self.dist_season.log_prob(x_part_obs["season"])            # logP(Season)
        log_p_part_obs["pneu"] = self.log_lik_class(x_part_obs, "pneu", ["season"])           # logP(Pneu | Season, Text)
        log_p_part_obs["inf"] = self.log_lik_class(x_part_obs, "inf", ["season"])             # logP(Inf | Season, Text)
        log_p_part_obs["dysp"] = torch.zeros(len(x_part_obs["dysp"]), device=self.device)
        log_p_part_obs["cough"] = torch.zeros(len(x_part_obs["cough"]), device=self.device)
        log_p_part_obs["nasal"] = torch.zeros(len(x_part_obs["nasal"]), device=self.device)

        # logP(Season, Pneu, Inf, Dysp, Cough, Nasal | Text) 
        # = logP(Season) + logP(Pneu | Season, Text) + logP(Inf | Season, Text)
        # + logP(Dysp | Pneu, Text) + logP(Cough | Pneu, Inf, Text) + logP(Nasal | Inf, Text)
        # concatenate fully observed and partially observed results
        # return losses for all classifiers separately, so they're easier to monitor
        log_p_individual = {key:torch.cat([l_fully_obs, log_p_part_obs[key]]) for key, l_fully_obs in log_p_fully_obs.items()}
        return log_p_individual
    
class DiscriminativeModelAbl(DiscriminativeModel):
    """
    DiscriminativeModelAbl: Bayesian network with discriminative text model, without relation between text and diagnoses 
    The conditional probability distributions for nodes dysp, cough and nasal are modeled by one text classifier
    per non-text parent configuration. The distributions P(season), P(pneu|season) and P(inf|season) are modeled as simple conditional
    probability tables, parameterized by one Bernoulli parameter per parent configuration. 

    n_emb: text embedding dimension
    hidden_dim: dict {var_name: [...]} containing list of hidden dimensions to use in classifiers of  "dysp", "cough" and "nasal" nodes
    dropout: dict {var_name: ...} containing dropout probability to use in classifiers of "dysp", "cough" and "nasal"
    seed: initialization seed 
    device: CPU/GPU

    Inherits methods from DiscriminativeModel class
    """
    
    def __init__(self, n_emb, hidden_dim, dropout, seed, device): 

        nn.Module.__init__(self)

        torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # initialize parameters
        self.params = nn.ParameterDict({
            "season": nn.Parameter(torch.rand(1)), # P(season=cold)
            "pneu": nn.Parameter(torch.rand(2)),   # P(pneu|season=warm), P(pneu|season=cold)
            "inf": nn.Parameter(torch.rand(2)),    # P(inf|season=warm), P(inf|season=cold)
        })
        self.classifiers = nn.ModuleDict({
            "dysp": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["dysp"], dropout["dysp"], seed, apply_sigmoid=False),     # P(dysp|pneu=no,text)
                                   TextEmbClassifier(n_emb, hidden_dim["dysp"], dropout["dysp"], seed, apply_sigmoid=False)]),     # P(dysp|pneu=yes,text)
            "cough": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),    # P(cough|pneu=no,inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),   # P(cough|pneu=no,inf=yes,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False),   # P(cough|pneu=yes,inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["cough"], dropout["cough"], seed, apply_sigmoid=False)]),   # P(cough|pneu=yes,inf=yes,text)
            "nasal": nn.ModuleList([TextEmbClassifier(n_emb, hidden_dim["nasal"], dropout["nasal"], seed, apply_sigmoid=False),    # P(nasal|inf=no,text)
                                    TextEmbClassifier(n_emb, hidden_dim["nasal"], dropout["nasal"], seed, apply_sigmoid=False)])    # P(nasal|inf=yes,text)
        })

        # create conditional probability distributions from Bernoulli parameters
        self.dist = {}
        self.update_dist()

        self.device = device

    def update_dist(self): 
        """ 
        Update Bernoulli distributions based on conditional parameters
        """
        
        # update distributions
        self.dist["season"] = Bernoulli(probs=torch.sigmoid(self.params["season"]))
        self.dist["pneu"] = Bernoulli(probs=torch.sigmoid(self.params["pneu"]))
        self.dist["inf"] = Bernoulli(probs=torch.sigmoid(self.params["inf"]))

    def log_p_cond(self, x, child, parents): 
        """ 
        Calculates P(child|parents) where parents can have arbitrary length, assuming each parent can take on 2 values 
        x: sample containing values for all variables
        child: name of child ("pneu", "inf", "dysp", "cough" or "nasal")
        parents: list of names of parents (e.g., for "cough" this is ["pneu", "inf"])
        returns: P(child|parents)
        """

        n = len(parents)

        x_child = x[child] 
        x_child = x_child[(...,) + (None,)*n]       # equivalent to x_child[:, None, None, ..., None] with number of None's equal to n
        prob = self.dist[child].log_prob(x_child)   # get prob P(Child=c|Parents) for all possible parent values
                                                    # shape (bs, (2,)*n), e.g. (bs, 2, 2, 2) for 3 parents
        for i in range(n): 
            parent = parents[i]
            x_parent = x[parent]
            x_parent = x_parent[(...,) + (None,)*(n-i)].expand(-1, 1, *(2,)*(n-i-1)) # shape (bs, 1, (2,)*(n-i-1))
            prob = torch.gather(prob, 1, x_parent.long()).squeeze(1) # select prob P(Child=c|Parent=p,Remaining parents)
                                                                     # shape (bs, (2,)*(n-i)) -> n-i parents remain         
        return prob
        
    def forward(self, x):
        """ 
        Calculate logP of sample x under current parameterization of the CPTs and text classifiers
        Combines logP(season,pneu,inf,dysp,cough,nasal|text) for fully observed portion of samples x 
            and logP(season,pneu,inf|text) for partially observed (= symptoms not observed) portion of samples x
        Is used to train all parameters in BN (so conditional probability table for season, and all 8 text classifiers)

        x: sample containing observed values for all variables, format {"season":tensor(), "pneu":tensor(), "inf":tensor(), "dysp": tensor(), "cough": tensor(), "nasal": tensor(), "text": tensor()} 
        returns: individual conditional log-probabilities for every variable, so they're easy to monitor separately during training
        """
        # update conditional probability table distributions with current parameters
        self.update_dist()
        
        # split batch into fully observed part and partially observed part 
        x_part_obs = {key:val[torch.isnan(x["dysp"])] for key, val in x.items()}
        x_fully_obs = {key:val[~torch.isnan(x["dysp"])] for key, val in x.items()}
    
        # fully observed
        log_p_fully_obs = {}
        log_p_fully_obs["season"] = self.dist["season"].log_prob(x_fully_obs["season"])      # logP(season)
        log_p_fully_obs["pneu"] = self.log_p_cond(x_fully_obs, "pneu", ["season"])           # logP(pneu|season)
        log_p_fully_obs["inf"] = self.log_p_cond(x_fully_obs, "inf", ["season"])             # logP(inf|season)
        log_p_fully_obs["dysp"] = self.log_lik_class(x_fully_obs, "dysp", ["pneu"])          # logP(dysp|pneu,text)
        log_p_fully_obs["cough"] = self.log_lik_class(x_fully_obs, "cough", ["pneu", "inf"]) # logP(cough|pneu,inf,text)
        log_p_fully_obs["nasal"] = self.log_lik_class(x_fully_obs, "nasal", ["inf"])         # logP(nasal|inf,text)

        # partially observed
        log_p_part_obs = {}
        log_p_part_obs["season"] = self.dist["season"].log_prob(x_part_obs["season"])        # logP(season)
        log_p_part_obs["pneu"] = self.log_p_cond(x_part_obs, "pneu", ["season"])             # logP(pneu|season)
        log_p_part_obs["inf"] = self.log_p_cond(x_part_obs, "inf", ["season"])               # logP(inf|Season)
        log_p_part_obs["dysp"] = torch.zeros(len(x_part_obs["dysp"]), device=self.device)
        log_p_part_obs["cough"] = torch.zeros(len(x_part_obs["cough"]), device=self.device)
        log_p_part_obs["nasal"] = torch.zeros(len(x_part_obs["nasal"]), device=self.device)

        # logP(Season, Pneu, Inf, Dysp, Cough, Nasal | Text) 
        # = logP(Season) + logP(Pneu | Season) + logP(Inf | Season)
        # + logP(Dysp | Pneu, Text) + logP(Cough | Pneu, Inf, Text) + logP(Nasal | Inf, Text)
        # concatenate fully observed and partially observed results
        # return losses for all classifiers separately, so they're easier to monitor
        log_p_individual = {key:torch.cat([l_fully_obs, log_p_part_obs[key]]) for key, l_fully_obs in log_p_fully_obs.items()}
        return log_p_individual