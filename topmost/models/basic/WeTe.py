import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

def to_list(data, device='cuda:0'):
    data_list = []
    for i in range(len(data)):
        idx = torch.where(data[i]>0)[0]
        data_list.append(torch.tensor([j for j in idx for _ in range(data[i,j])], device=device))
    return data_list

def to_list_parallel(data: torch.Tensor):
    # data = data.to(device)
    B, N = data.shape

    # Flatten the data and keep track of batch indices
    flat_data = data.view(-1)
    nonzero_mask = flat_data > 0
    flat_data = flat_data[nonzero_mask]

    # Compute corresponding batch and index positions
    indices = torch.arange(B * N, device=data.device)[nonzero_mask]
    batch_ids = indices // N
    feature_ids = indices % N

    # Repeat batch and feature indices based on count in flat_data
    repeated_batch_ids = torch.repeat_interleave(batch_ids, flat_data)
    repeated_feature_ids = torch.repeat_interleave(feature_ids, flat_data)

    # Use scatter to collect per batch
    output = [repeated_feature_ids[repeated_batch_ids == i] for i in range(B)]
    return output

###############################################################################
# Weibull-based encoder network (similar to Infer_Net)
###############################################################################
class WeibullEncoder(nn.Module):
    """
    Weibull inference network for topic proportion
    """
    def __init__(self, vocab_size=2000, d_hidden=256, num_topics=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.num_topics = num_topics

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 2 * self.num_topics),
            nn.Softplus()  # outputs must be >= 0
        )

    def reparameterize(self, wei_shape, wei_scale, sample_num=5):
        """
        Weibull reparameterization trick
        :param wei_shape: (batch, K)
        :param wei_scale: (batch, K)
        :param sample_num: number of samples to average
        """
        # sample uniform noise
        eps = torch.rand(
            sample_num, wei_shape.size(0), wei_shape.size(1),
            device=wei_shape.device
        )
        # Weibull sampling
        # theta = scale * (-log(u))^(1/shape)
        # repeat scale/shape across sample_num dimension
        scale_ = wei_scale.unsqueeze(0).repeat(sample_num, 1, 1)
        shape_ = wei_shape.unsqueeze(0).repeat(sample_num, 1, 1)
        theta_samples = scale_ * torch.pow(-torch.log(eps+1e-10), 1.0 / shape_)

        # clamp to avoid numerical instabilities, then average over samples
        theta_samples = torch.clamp(theta_samples, min=1e-10, max=100.0)
        theta = torch.mean(theta_samples, dim=0)
        return theta

    def forward(self, bow_x):
        """
        Given a BoW input, return unnormalized topic proportions (theta)
        :param bow_x: (batch, vocab_size)
        """
        out = self.encoder(bow_x)          # (batch, 2*K)
        wei_shape, wei_scale = torch.chunk(out, 2, dim=-1)
        # clamp shape/scale for numerical stability
        wei_shape = torch.clamp(wei_shape, 0.1, 100.0)
        wei_scale = torch.clamp(wei_scale, 1e-4, 1e4)
        theta = self.reparameterize(wei_shape, wei_scale, sample_num=5)
        return theta


###############################################################################
# Main WeTe model, refactored to a single class in the style of NeuroMax
###############################################################################
class WeTe(nn.Module):
    """
    WeTe: Representing Mixtures of Word Embeddings with Mixtures of Topic Embeddings
    Refactored to match the style of NeuroMax code.

    Forward pass returns a dictionary of losses, similar to how NeuroMax does.
    """
    def __init__(
        self,
        vocab_size,           # V
        num_topics=100,       # K
        embedding_dim=200,    # dimension of word/topic embeddings
        hidden_dim=256,       # dimension of encoder's hidden layers
        beta=0.5,             # weight for forward vs. backward cost
        epsilon=1.0,          # weight for Poisson-likelihood term
        pretrained_WE=None,   # optional pretrained word embeddings
        device='cuda',
        init_alpha=False,     # whether to init topics from K-means or random
        glove_path=None,      # path to glove embeddings, if needed
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.beta = beta
        self.epsilon = epsilon
        self.real_min = 1e-30
        self.init_alpha = init_alpha
        self.device = device

        ########################################
        # 1) Embedding layers for topics & words
        ########################################
        # topic embeddings [K, embedding_dim]
        self.topic_layer = nn.Embedding(self.num_topics, self.embedding_dim)
        # word embeddings [V, embedding_dim]
        self.word_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # In typical WeTe, either:
        # (a) load pretrained word embeddings from GloVe, or
        # (b) init from normal distribution
        self.init_topic(pretrained_WE, glove_path)

        # Keep track of row indices for topic & word
        self.topic_id = torch.arange(self.num_topics, device=self.device).unsqueeze(-1)
        self.word_id = torch.arange(self.vocab_size, device=self.device).unsqueeze(-1)

        # We'll store these after calling update_embeddings()
        self.rho = None   # word embeddings
        self.alpha = None # topic embeddings

        ########################################
        # 2) The encoder for doc-level topic props
        ########################################
        self.InferNet = WeibullEncoder(
            vocab_size=self.vocab_size,
            d_hidden=self.hidden_dim,
            num_topics=self.num_topics
        )

        # Move everything to the proper device
        self.to(device)
        self.update_embeddings()


    def init_topic(self, pretrained_WE, glove=None):
        """
        Either load pretrained embeddings (e.g., from GloVe) or
        initialize from a random distribution. Possibly do K-means for topic init.
        """
        if pretrained_WE is not None:
            # Provided as a numpy array from outside
            word_e = torch.from_numpy(pretrained_WE).float()
            print("Loaded pretrained word embeddings from array:", word_e.shape)
            with torch.no_grad():
                self.word_layer.weight.copy_(word_e)
            self.word_layer.weight.requires_grad = False  # freeze or finetune as you wish
        else:
            if glove is not None:
                # If user wants to do file-based GloVe load
                print(f"Load pretrained glove embeddings from : {glove}")
                word_e = np.random.rand(self.vocab_size, self.embedding_dim)*0.01
                # A simple example loop (not fully robust for real GloVe files):
                with open(glove, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        vals = line.strip().split()
                        if len(vals) == self.embedding_dim+1:
                            word = vals[0]
                            # ... match word in self.voc ... (requires a dictionary)
                            # not fully shown here
                            # if matched: place embedding
                            pass
                # Convert to torch
                word_e = torch.tensor(word_e, dtype=torch.float32)
                with torch.no_grad():
                    self.word_layer.weight.copy_(word_e)
                self.word_layer.weight.requires_grad = False  # freeze or finetune
            else:
                # random normal initialization
                print("Initialize word embedding from N(0, 0.02)")
                nn.init.normal_(self.word_layer.weight, mean=0.0, std=0.02)

        # For topics:
        if self.init_alpha:
            # If you want to do k-means on word embeddings to initialize topics
            # (dummy placeholder below)
            print("Initialize topic embeddings with K-means on word embeddings (placeholder).")
            nn.init.normal_(self.topic_layer.weight, mean=0.0, std=0.5)
        else:
            # random normal
            print("Initialize topic embeddings from N(0, 0.5).")
            nn.init.normal_(self.topic_layer.weight, mean=0.0, std=0.5)


    def update_embeddings(self):
        """
        Retrieve the current word (rho) and topic (alpha) embeddings
        from embedding layers, shaped:
          rho:   [vocab_size, embedding_dim]
          alpha: [num_topics, embedding_dim]
        """
        self.rho = self.word_layer(self.word_id).squeeze()     # (V, h)
        self.alpha = self.topic_layer(self.topic_id).squeeze() # (K, h)


    def cal_phi(self):
        """
        phi_{vk} = softmax(rho_v^T alpha_k, dim=0 over v)
        => returns [vocab_size, num_topics]
        """
        inner_p = torch.matmul(self.rho, self.alpha.t())  # (V, K)
        phi = F.softmax(inner_p, dim=0)
        return phi


    def cost_ct(self, inner_p, x, theta):
        """
        Compute forward_cost & backward_cost as in WeTe paper.
        :param inner_p: (V, K) = rho x alpha^T
        :param x: list of lists of word indices for each document: shape [batch of docs]
        :param theta: (batch, K)
        """
        # cost_c = exp( - inner_p ), shape (V,K)
        cost_c = torch.exp(-inner_p).clamp(self.real_min, 1e10)

        forward_cost = 0.0
        backward_cost = 0.0
        # normalize doc-level topic proportions
        theta_norm = F.softmax(theta, dim=-1)  # (batch,K)

        # We'll need the "dis_d" which is exp(inner_p) or similar
        dis_d = torch.clamp(torch.exp(inner_p), self.real_min, 1e10)

        # Each doc is a set of word indices in x
        # so for doc j, x_j is something like [w1, w2, ...].
        for doc_indices, doc_theta in zip(x, theta_norm):
            # shape: doc_indices is e.g. (num_words_in_doc,)
            # forward plan: doc_dis_forw = dis_d[doc_indices] * doc_theta
            # doc_dis_forw => shape [N_j, K]
            forward_doc_dis = dis_d[doc_indices] * doc_theta.unsqueeze(0)
            sum_over_topic = forward_doc_dis.sum(dim=1, keepdim=True) + self.real_min
            forward_pi = forward_doc_dis / sum_over_topic
            # forward cost accumulates over doc words
            forward_cost += (cost_c[doc_indices] * forward_pi).sum(dim=1).mean()

            # backward plan
            doc_dis_back = dis_d[doc_indices]  # (N_j, K)
            sum_over_word = doc_dis_back.sum(dim=0, keepdim=True) + self.real_min
            backward_pi = doc_dis_back / sum_over_word
            # sum( cost_c * backward_pi ) over words -> shape(K,)
            backward_cost_doc = (cost_c[doc_indices] * backward_pi).sum(dim=0)
            # multiply by doc_theta -> scalar
            backward_cost_doc = (backward_cost_doc * doc_theta).sum()
            backward_cost += backward_cost_doc

        return forward_cost, backward_cost


    def Poisson_likelihood(self, x_bow, re_x):
        """
        Negative log of Poisson likelihood
        :param x_bow: (batch, vocab_size)
        :param re_x:  (vocab_size, batch) => we usually transpose it
        """
        # re_x + 1e-10 for numerical stability
        # formula: -( x * log(lambda) - lambda - lgamma(x+1) ).sum(dim=-1).mean()
        return -(x_bow * torch.log(re_x + 1e-10) - re_x - torch.lgamma(x_bow + 1.0)
                 ).sum(dim=-1).mean()


    def forward(self, input_dict, epoch_id=None, batch_idx=None):
        """
        Forward pass that returns a dict of losses, similar to NeuroMax.

        :param input_dict: a dictionary containing
           "x": list of docs, each is a list of word indices
           "bow": (batch_size, vocab_size) bag-of-words
        :param epoch_id: optional, not strictly needed

        returns dict with { 'loss', 'loss_TM', 'loss_forward', 'loss_backward' }
        """
        # 1) Extract inputs
        # x_docs = input_dict["contextual_embed"]       # e.g. list of arrays for doc word indices
        bow = input_dict["data"]        # shape (batch_size, vocab_size)
        # print(bow.to(torch.int))
        x_docs = to_list_parallel(bow.to(torch.int))

        # 2) Infer doc-topic proportions via Weibull encoder
        theta_unnorm = self.InferNet(bow)  # shape (batch, K)

        # 3) Update alpha, rho from embeddings
        self.update_embeddings()

        # 4) Calculate the forward/backward transport cost
        #    inner_p = (V,K), where V= vocab_size, K= num_topics
        inner_p = torch.matmul(self.rho, self.alpha.t())  # (V,K)
        forward_cost, backward_cost = self.cost_ct(inner_p, x_docs, theta_unnorm)

        # 5) Compute the doc→word reconstruction
        #    phi => (V,K), theta => (batch,K)
        #    re_x => (V, batch)
        phi = self.cal_phi()
        re_x = torch.matmul(phi, theta_unnorm.t())  # (V, batch)
        # Poisson negative log-likelihood => sum over vocab, average over batch
        TM_cost = self.Poisson_likelihood(bow, re_x.t())

        # 6) Weighted sum of forward/backward + Poisson => final loss
        #    Typically from the WeTe paper: loss = beta * forward + (1-beta)*backward + epsilon*TM
        #    but one can interpret "beta" in multiple ways.
        total_loss = self.beta * forward_cost \
                     + (1.0 - self.beta) * backward_cost \
                     + self.epsilon * TM_cost

        # 7) Return a dictionary of losses
        rst_dict = {
            'loss': total_loss,
            'loss_TM': TM_cost,
            'loss_CT_forward': forward_cost,
            'loss_CT_backward': backward_cost,
        }
        return rst_dict

    def save_embeddings(self, path='out.pkl'):
        """
        Save word & topic embeddings for later inspection.
        """
        # Move to CPU for pickle
        word_e = self.rho.detach().cpu().numpy()
        topic_e = self.alpha.detach().cpu().numpy()
        with open(path, 'wb') as f:
            pickle.dump([word_e, topic_e], f)
        print(f"Saved embeddings to {path}")
