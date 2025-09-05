#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict


from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict

@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerClusteredFedACG(ServerM):
    def __init__(self, args):
        super().__init__(args)
        self.num_clusters = int(self.args.server.num_clusters)
        self.clustering_interval = int(self.args.server.clustering_interval)

        self.warmup_rounds = self.args.server.get('warmup_rounds', 300)
    
    def set_momentum(self, model):
        self.client_clusters = {i: 0 for i in range(self.args.trainer.num_clients)}

        zero_momentum = copy.deepcopy(model.state_dict())
        for key in zero_momentum.keys():
            zero_momentum[key] = torch.zeros_like(zero_momentum[key])
            
        self.cluster_momentums = {
            i: copy.deepcopy(zero_momentum) for i in range(self.num_clusters)
        }
        
        def flatten_state_dict(sd):
            return torch.cat([p.flatten() for p in sd.values()])
            
        self.client_update_history = {}
        self._flatten_state_dict = flatten_state_dict

    def _update_clusters(self):
        
        clients_to_cluster = list(self.client_update_history.keys())
        if len(clients_to_cluster) < self.num_clusters:
            return

        update_vectors = [self.client_update_history[cid].cpu().numpy() for cid in clients_to_cluster]
        update_matrix = np.vstack(update_vectors)

        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10).fit(update_matrix)
            new_labels = kmeans.labels_

            for i, client_id in enumerate(clients_to_cluster):
                self.client_clusters[client_id] = new_labels[i]
            

    def _alignment_gate(self, client_id: int, cluster_id: int):
        alpha_max = float(self.args.server.get('lookahead_alpha', 1.0))
        t = float(self.args.server.get('gate_threshold', 0.0))
        p = float(self.args.server.get('gate_power', 1.0))
        cold = float(self.args.server.get('gate_cold_start', 0.0))

        if client_id not in self.client_update_history:
            return alpha_max * cold

        delta_vec = self.client_update_history[client_id]
        mom_vec = self._flatten_state_dict(self.cluster_momentums[cluster_id])
        sim = F.cosine_similarity(delta_vec, mom_vec, dim=0, eps=1e-8).item()

        if sim <= t:
            gate = 0.0
        else:
            gate = ((sim - t) / (1.0 - t)) ** p

        return alpha_max * gate

    @torch.no_grad()
    def FedACG_lookahead(self, model, client_id):
        cluster_id = self.client_clusters.get(client_id, 0)
        cluster_momentum = self.cluster_momentums[cluster_id]
    
        alpha = self._alignment_gate(client_id, cluster_id)
    
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in cluster_momentum.keys():
            sending_model_dict[key] += alpha * cluster_momentum[key]
    
        lookahead_model = copy.deepcopy(model)
        lookahead_model.load_state_dict(sending_model_dict)
        return lookahead_model
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        """
        聚合模型更新，并为每个集群分别更新动量。
        """
        C = len(client_ids)
        
        for i, cid in enumerate(client_ids):
            client_delta_dict = {k: v[i] for k, v in local_deltas.items()}
            self.client_update_history[cid] = self._flatten_state_dict(client_delta_dict)

        cluster_deltas_agg = defaultdict(lambda: {k: torch.zeros_like(v[0]) for k, v in local_deltas.items()})
        cluster_client_counts = defaultdict(int)

        for i, cid in enumerate(client_ids):
            cluster_id = self.client_clusters.get(cid, 0)
            cluster_client_counts[cluster_id] += 1
            for key in local_deltas:
                cluster_deltas_agg[cluster_id][key] += local_deltas[key][i]

        for cid, agg_delta in cluster_deltas_agg.items():
            avg_cluster_delta = {k: v / cluster_client_counts[cid] for k, v in agg_delta.items()}
            
            for key in self.cluster_momentums[cid]:
                self.cluster_momentums[cid][key] = self.args.server.momentum * self.cluster_momentums[cid][key] + avg_cluster_delta[key]
        
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key]) / C
            
        return local_weights

