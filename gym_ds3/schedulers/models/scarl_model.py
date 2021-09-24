import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from platform import machine

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Ref: https://github.com/ita9naiwa/SCARL-implementation

def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = 0.001#np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        try:
            m.bias.data.normal_(0.0, 0.0001)
        except:
            pass


class mlp_layer(nn.Module):
    def __init__(self, input_size, output_size, activation='tanh', drouput_prob=0.0):
        super(mlp_layer, self).__init__()

        self.affine = nn.Linear(input_size, output_size)
        #self.dropout = nn.Dropout(drouput_prob)
        weight_init(self.affine)

        if activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'relu':
            self.activation = F.relu()

    def forward(self, x):
        x = self.activation(self.affine(x))
        #return self.dropout(x)
        return x

class custom_embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(custom_embedding, self).__init__()        
        self.affine1 = mlp_layer(input_size, output_size)
        self.affine2 = mlp_layer(output_size, output_size)
        weight_init(self.affine1)
        weight_init(self.affine2)
        
    def forward(self, x):
        return self.affine2(self.affine1(x))

nl_emb = custom_embedding



class Attention(nn.Module):
    def __init__(self, hidden_size, query_size, use_softmax=False):
        super(Attention, self).__init__()

        self.use_softmax = use_softmax
        self.W_query = nn.Linear(query_size, hidden_size, bias=True)
        self.W_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        V = torch.normal(torch.zeros(hidden_size), 0.0001)
        self.V = nn.Parameter(V)
        weight_init(V)
        weight_init(self.W_query)
        weight_init(self.W_ref)

    def forward(self, query, ref):
        """
        Args:
            query: [hidden_size]
            ref:   [seq_len x hidden_size]
        """

        seq_len  = ref.size(0)
        query = self.W_query(query) # [hidden_size]

        _ref = self.W_ref(ref)  # [seq_len x hidden_size]
        #V = self.V # [1 x hidden_size]
        m = torch.tanh(query + _ref)

        logits = torch.matmul(m, self.V)
        if self.use_softmax:
            logits = torch.softmax(logits, dim=0)
        else:
            logits = logits

        return logits


class query_att(nn.Module):
    def __init__(self, hidden_size, query_size, use_softmax=False, as_sum=True):
        super(query_att, self).__init__()

        self.attention = Attention(hidden_size, query_size, use_softmax=True)


    def forward(self, query, ref):
        """
        Args:
            query: [hidden_size]
            ref:   [seq_len x hidden_size]
        """

        softmax_res = self.attention(query, ref)
        ret = torch.matmul(softmax_res, ref)
        return torch.matmul(softmax_res, ref)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dim = None
    def forward(self, q, k, v, mask=None):
        if self.dim is None:
            self.dim = 1 / np.sqrt(q.size(-1))

        attn = torch.mm(q, k.transpose(-2, -1))
        attn = attn / self.temperature
        attn = self.dim * attn

        attn = self.softmax(attn)
        #attn = self.dropout(attn)
        output = torch.mm(attn, v)

        return output, attn

class merge_layer(nn.Module):
    def __init__(self, hidden_size, output_size, activation='tanh'):
        super(merge_layer, self).__init__()

        self.mlp_layer = mlp_layer(hidden_size, output_size, activation=activation)

    def forward(self, x):
        x = torch.mean(x, dim=-1)
        return self.mlp_layer(x)



class Pointer(nn.Module):
    def __init__(self,
            input_size,
            dec_input_size,
            embedding_size,
            hidden_size,
            n_glimpses,
            tanh_exploration,
            halt_action=False):
        super(Pointer, self).__init__()

        self.input_size = input_size
        self.dec_input_size = dec_input_size
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.halt_action = halt_action
        self.dec_input_embedding = custom_embedding(self.dec_input_size, embedding_size)

        self.embedding = custom_embedding(input_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        if self.halt_action:
            self.halt = nn.Parameter(torch.FloatTensor(embedding_size))
            self.halt.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size)
        self.glimpse = Attention(hidden_size)

        self.criterion = nn.CrossEntropyLoss()

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def encode(self, inputs):
        inputs_dim = inputs.dim()
        batch_size = inputs.size(0)
        if inputs_dim == 3:
            job_size = inputs.size(1)
            seq_len = inputs.size(2)
            embedded = self.embedding(inputs)
            if self.halt_action:
                stop = self.halt.unsqueeze(0).unsqueeze(0).repeat(1, batch_size, 1)
                embedded = torch.cat([embedded, stop], 1)

        if inputs_dim == 2:
            embedded = stop = self.halt.unsqueeze(0).unsqueeze(0).repeat(1, batch_size, 1)
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        return encoder_outputs, (hidden, context)

    def decode(self, encoder_outputs, decoder_input=None, hc=None):
        batch_size = encoder_outputs.size(0)
        if decoder_input is None:
            decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)

        _, (hidden, context) = self.decoder(decoder_input, hc)
        query = hidden.squeeze(0)
        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(query, encoder_outputs)
            query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

        _, logits = self.pointer(query, encoder_outputs)
        return logits


    def loss(self, logits, targets):
        loss = self.criterion(logits, target)
        return loss

class pointer_scheduler(nn.Module):
    def __init__(self,
        job_size,
        machine_size,
        embedding_size,
        dec_input_size,
        hidden_size,
        n_glimpses):
        super(pointer_scheduler, self).__init__()

        self.job_size, self.machine_size = job_size, machine_size

        self.job_pointer = Pointer(
            job_size, dec_input_size, embedding_size, hidden_size,
            n_glimpses=0, tanh_exploration=10, halt_action=False)

        self.machine_pointer = Pointer(
            machine_size, dec_input_size, embedding_size, hidden_size,
            n_glimpses=0, tanh_exploration=10, halt_action=False)
    def select_job(self, jobs, job_encoded=None, job_hc=None):
        if job_encoded is None or job_hc is None:
            job_encoded, job_hc = self.job_pointer.encode(jobs)

        job_logit = self.job_pointer.decode(
            job_encoded,
            decoder_input=None,
            hc=job_hc)


        job_prob = torch.softmax(job_logit, dim=1)
        job_sampler = Categorical(job_prob)
        j = job_sampler.sample()
        return job_encoded[:, j, :], j.detach().numpy(), job_sampler.log_prob(j)

    def select_machine(self, job_encoded, machines):
        machine_encoded, machine_hc = self.machine_pointer.encode(machines)
        machine_logit = self.machine_pointer.decode(
            machine_encoded,
            decoder_input=job_encoded,
            hc=machine_hc)
        machine_prob = torch.softmax(machine_logit, dim=1)
        machine_sampler = Categorical(machine_prob)
        m = machine_sampler.sample()
        return m.detach().numpy(), machine_sampler.log_prob(m)

    def forward(self, jobs, machines):
        job_encoded, job_hc = self.job_pointer.encode(jobs)
        machine_encoded, machine_hc = self.machine_pointer.encode(machines)


        job_logit = self.job_pointer.decode(
            job_encoded,
            decoder_input=machine_hc[0],
            hc=job_hc)


        job_prob = torch.softmax(job_logit, dim=1)
        job_sampler = Categorical(job_prob)
        j = job_sampler.sample()

        machine_logit = self.machine_pointer.decode(
            machine_encoded,
            decoder_input=job_encoded[:, j, : ],
            hc=machine_hc)

        machine_prob = torch.softmax(machine_logit, dim=1)
        machine_sampler = Categorical(machine_prob)
        m = machine_sampler.sample()
        return (j.detach().numpy(), m.detach().numpy()), job_sampler.log_prob(j) + machine_sampler.log_prob(m)


    def get_action(self, jobs, machines):
        job_prob, machine_prob = self.forward(jobs, machines)
        jcc, mcc = Categorical(job_prob), Categorical(machine_prob)
        js = jcc.sample()
        ms = mcc.sample()
        logpas = jcc.log_prob(js) + mcc.log_prob(ms)
        return jcc.sample().detach().numpy(), mcc.sample().detach().numpy(), logpas


class EmbeddingLayer(nn.Module):
    def __init__(self, job_dim, machine_dim, embedding_size=16):
        super(EmbeddingLayer, self).__init__()
        self.job_dim = job_dim
        self.machine_dim = machine_dim
        self.embedding_size = embedding_size

        self.job_embedding = custom_embedding(
            self.job_dim, self.embedding_size)

        self.machine_embedding = custom_embedding(
            self.machine_dim, self.embedding_size)

    def forward(self, jobs, machines):
        job_embedding = self.job_embedding(jobs)
        machine_embedding = self.machine_embedding(machines)
        return job_embedding, machine_embedding


class LSTMEncoder(nn.Module):
    def __init__(self, job_dim, machine_dim, embedding_size=16):
        super(LSTMEncoder, self).__init__()
        self.job_dim = job_dim
        self.machine_dim = machine_dim
        self.embedding_size = embedding_size
        self.job_lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        self.machine_lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)

    def forward(self, job_embedding, machine_embedding):
        _, (job_global, _) = self.job_lstm(job_embedding.unsqueeze(0))
        _, (machine_global, _) = self.job_lstm(machine_embedding.unsqueeze(0))
        return (job_global[0, 0], machine_global[0, 0])

class MLPEncoder(nn.Module):
    def __init__(self, job_dim, machine_dim, embedding_size=16):
        super(MLPEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.job_summary1 = custom_embedding(
            self.embedding_size, self.embedding_size)

        self.job_summary2 = custom_embedding(
            self.embedding_size, self.embedding_size)

        self.machine_summary1 = custom_embedding(
            self.embedding_size, self.embedding_size)

        self.machine_summary2 = custom_embedding(
            self.embedding_size, self.embedding_size)

    def forward(self, job_embedding, machine_embedding):
        job_global = self.job_summary2(torch.sum(self.job_summary1(job_embedding), -2))
        machine_global = self.machine_summary2(torch.sum(self.machine_summary1(machine_embedding), -2))
        return job_global, machine_global

class AttentionLogit(nn.Module):
    def __init__(self, embedding_size=32):
        self.embedding_size = embedding_size
        super(AttentionLogit, self).__init__()

        self.job_merge = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.machine_merge = nn.Linear(3 * self.embedding_size, self.embedding_size)

        self.job_attention = Attention(self.embedding_size, use_softmax=False)
        self.machine_attention = Attention(self.embedding_size, use_softmax=False)

    def get_job_logit(self, job_repr, machine_repr, job_global, machine_global):
        global_repr = torch.cat((job_global, machine_global)).unsqueeze(0)
        global_repr = self.job_merge(global_repr)
        _, job_logit = self.job_attention(global_repr, job_repr.unsqueeze(0))

        job_logit = job_logit.squeeze(0)
        return job_logit

    def get_machine_logit(self, selected_job, machine_repr, job_global, machine_global):
        global_repr = torch.cat((selected_job, job_global, machine_global)).unsqueeze(0)
        global_repr = self.machine_merge(global_repr)
        _, machine_logit = self.machine_attention(global_repr, machine_repr.unsqueeze(0))
        machine_logit = machine_logit.squeeze(0)
        return machine_logit


class attention_net(nn.Module):
    def __init__(self,
        num_tasks,
        num_jobs,
        num_actions,
        device,
        job_dim,
        machine_dim,
        embedding_size=16,
        N=0,
        wo_attention=False):

        super(attention_net, self).__init__()

        self.num_tasks = num_tasks
        self.num_jobs = num_jobs
        self.num_actions = num_actions
        self.device = device

        self.num_tasks_in_jobs = self.num_tasks // self.num_jobs

        self.hmm_size = 15 # task_obs.shape[-1]

        self.job_dim = job_dim
        self.machine_dim = machine_dim
        self.embedding_size = embedding_size
        self.N = N
        self.wo_attention = wo_attention
        if wo_attention:
            self.N = 0
        self.j_emb = nl_emb(self.hmm_size, embedding_size)
        self.m_emb = nl_emb(machine_dim, embedding_size)

        self.j_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]
        self.m_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]
        if self.wo_attention is False:
            self.att_to_j = query_att(embedding_size, embedding_size)
            self.att_to_m = query_att(embedding_size, embedding_size)

        self.j_att = Attention(embedding_size, 3 * embedding_size, False)
        self.m_att = Attention(embedding_size, 3 * embedding_size, False)
        self.last_j = torch.normal(torch.zeros(self.embedding_size), 0.0001)
        self.ent1 = 0
        self.c1 = 0
        self.ent2 = 0
        self.c2 = 0
        self.ent3 = 0
        self.c3 = 0
        self.ent4 = 0
        self.c4 = 0
        self.test = False
        no_select_job = torch.normal(torch.zeros(self.embedding_size), 0.0001)
        self.no_select_job = nn.Parameter(no_select_job)

        no_select_machine = torch.normal(torch.zeros(self.embedding_size), 0.0001)
        self.no_select_machine = nn.Parameter(no_select_machine)

    def reset(self,):
        self.last_j = torch.zeros(self.embedding_size)

    def get_job_attention(self, input, query):
        if self.wo_attention is True:
            return torch.mean(query, dim=0)
        else:
            return self.att_to_j(input, query)


    def get_node_attention(self, input, query):
        if self.wo_attention is True:
            return torch.mean(query, dim=0)
        else:
            return self.att_to_m(input, query)

    def get_embedding(self, jobs, machines):
        j = self.j_emb(jobs)
        m = self.m_emb(machines)
        return j, m

    def forward(self, jobs, machines, allocable_jobs=None, allocable_machines=None, argmax=False):
        job_input_size = jobs.size(0)
        machine_input_size = machines.size(0)

        E_j, E_m = self.get_embedding(jobs, machines)
        g_j1 = self.get_job_attention(self.last_j, E_j)
        g_m1 = self.get_node_attention(self.last_j, E_m)
        E_j = torch.cat([E_j, self.no_select_job.unsqueeze(0)], dim=0)
        E_m = torch.cat([E_m, self.no_select_machine.unsqueeze(0)], dim=0)

        g_1 = torch.cat([self.last_j, g_j1, g_m1])
        j_logits = self.j_att(g_1, E_j)

        ### selecting processes
        if allocable_jobs is not None:
            x = []
            for _ in range(job_input_size):
                if _ not in allocable_jobs:
                    x.append(_)
            if len(x) > 0:
                mask = torch.from_numpy(np.array(x, dtype=int))

                j_logits[mask] = -1e8

        job_softmax = torch.softmax(j_logits, 0)
        job_sampler = Categorical(job_softmax)

        if argmax:
            selected_job = torch.argmax(job_softmax)
        else:
            try:
                selected_job = job_sampler.sample()
            except:
                raise UnboundLocalError;

        # if selected_job == job_input_size:
        #     return -1, -1, job_sampler.log_prob(selected_job)

        as_i = int(selected_job.detach().numpy())

        e_js = E_j[selected_job]
        g_j2 = self.get_job_attention(e_js, E_j)
        g_m2 = self.get_node_attention(e_js, E_m)
        g_2 = torch.cat([e_js, g_j2, g_m2])
        m_logits = self.m_att(g_2, E_m)
        self.last_j = e_js.detach()
        ### selecting process
        if allocable_machines is not None:
            x = []
            for _ in range(machine_input_size):
                if _ not in allocable_machines[as_i]:
                    x.append(_)
            mask = torch.from_numpy(np.array(x, dtype=int))
            m_logits[mask] = -1e8

        machine_softmax = torch.softmax(m_logits, 0)
        machine_sampler = Categorical(machine_softmax)
        if argmax:
            selected_machine = torch.argmax(machine_softmax, -1)
        else:
            selected_machine = machine_sampler.sample()


        logpas = machine_sampler.log_prob(selected_machine) + job_sampler.log_prob(selected_job)
        if selected_machine == machine_input_size:
            return -1, -1, logpas
        return int(selected_job.detach().numpy()), int(selected_machine.detach().numpy()), logpas

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to(device)



class attention_net_noc(nn.Module):
    def __init__(self,
        job_dim,
        machine_dim,
        embedding_size=16,
        N=0,
        wo_attention=False):

        super(attention_net_noc, self).__init__()

        self.job_dim = job_dim
        self.machine_dim = machine_dim
        self.embedding_size = embedding_size
        self.N = N
        self.wo_attention = wo_attention
        if wo_attention:
            self.N = 0

        self.j_emb = nl_emb(job_dim, embedding_size)
        self.m_emb = nl_emb(machine_dim, embedding_size)

        self.j_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]
        self.m_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]

        self.g1_merge = mlp_layer(3 * embedding_size, embedding_size)
        self.g2_merge = mlp_layer(3 * embedding_size, embedding_size)

        if self.wo_attention is False:
            self.att_to_j = query_att(embedding_size, embedding_size)
            self.att_to_m = query_att(embedding_size, embedding_size)

        self.j_att = Attention(embedding_size, embedding_size, False)
        self.m_att = Attention(embedding_size, embedding_size, False)
        self.last_j = torch.normal(torch.zeros(self.embedding_size), 0.0001)
        self.ent1 = 0
        self.c1 = 0
        self.ent2 = 0
        self.c2 = 0
        self.ent3 = 0
        self.c3 = 0
        self.ent4 = 0
        self.c4 = 0
        self.test = False

    def reset(self,):
        self.last_j = torch.zeros(self.embedding_size)

    def get_job_attention(self, input, query):
        if self.wo_attention is True:
            return torch.mean(query, dim=0)
        else:
            return self.att_to_j(input, query)


    def get_node_attention(self, input, query):
        if self.wo_attention is True:
            return torch.mean(query, dim=0)
        else:
            return self.att_to_m(input, query)

    def get_embedding(self, jobs, machines):
        j = self.j_emb(jobs)
        m = self.m_emb(machines)
        return j, m

    def forward(self, jobs, machines, allocable_jobs=None, allocable_machines=None, argmax=False):
        job_input_size = jobs.size(0)
        machine_input_size = machines.size(0)

        E_j, E_m = self.get_embedding(jobs, machines)
        g_j1 = self.get_job_attention(self.last_j, E_j)
        g_m1 = self.get_node_attention(self.last_j, E_m)

        g_1 = torch.cat([self.last_j, g_j1, g_m1])
        g_1 = self.g1_merge(g_1)
        j_logits = self.j_att(g_1, E_j)

        ### selecting processes
        if allocable_jobs is not None:
            x = []
            for _ in range(job_input_size):
                if _ not in allocable_jobs:
                    x.append(_)
            if len(x) > 0:
                mask = torch.from_numpy(np.array(x, dtype=int))
                j_logits[mask] = -1e8

        job_softmax = torch.softmax(j_logits, 0)
        job_sampler = Categorical(job_softmax)


        if argmax:
            selected_job = torch.argmax(job_softmax)
        else:
            try:
                selected_job = job_sampler.sample()
            except:
                raise UnboundLocalError;


        as_i = int(selected_job.detach().numpy())

        e_js = E_j[selected_job]
        g_j2 = self.get_job_attention(e_js, E_j)
        g_m2 = self.get_node_attention(e_js, E_m)
        g_2 = torch.cat([e_js, g_j2, g_m2])
        g_2 = self.g2_merge(g_2)
        m_logits = self.m_att(g_2, E_m)
        self.last_j = e_js.detach()
        ### selecting process
        if allocable_machines is not None:
            x = []
            for _ in range(machine_input_size):
                if _ not in allocable_machines[as_i]:
                    x.append(_)
            mask = torch.from_numpy(np.array(x, dtype=int))
            m_logits[mask] = -1e8

        machine_softmax = torch.softmax(m_logits, 0)
        #machine_softmax[machine_softmax < 0] = 0
        machine_sampler = Categorical(machine_softmax)
        if argmax:
            selected_machine = torch.argmax(machine_softmax, -1)
        else:
            selected_machine = machine_sampler.sample()


        logpas = machine_sampler.log_prob(selected_machine) + job_sampler.log_prob(selected_job)
        return (int(selected_job.detach().numpy()), int(selected_machine.detach().numpy())), logpas


class attention_net_nogr(nn.Module):
    def __init__(self,
        job_dim,
        machine_dim,
        embedding_size=16,
        N=0,
        wo_attention=False):

        super(attention_net_nogr, self).__init__()

        self.job_dim = job_dim
        self.machine_dim = machine_dim
        self.embedding_size = embedding_size
        self.N = N
        self.wo_attention = wo_attention
        if wo_attention:
            self.N = 0

        self.j_emb = nl_emb(job_dim, embedding_size)
        self.m_emb = nl_emb(machine_dim, embedding_size)

        self.j_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]
        self.m_ll = [mlp_layer(embedding_size, embedding_size) for _ in range(N)]

        self.j_att = Attention(embedding_size, embedding_size, False)
        self.m_att = Attention(embedding_size, embedding_size, False)
        self.last_j = torch.normal(torch.zeros(self.embedding_size), 0.0001)
        self.ent1 = 0
        self.c1 = 0
        self.ent2 = 0
        self.c2 = 0
        self.ent3 = 0
        self.c3 = 0
        self.ent4 = 0
        self.c4 = 0
        self.test = False

    def reset(self,):
        self.last_j = torch.zeros(self.embedding_size)

    def get_embedding(self, jobs, machines):
        j = self.j_emb(jobs)
        m = self.m_emb(machines)
        return j, m

    def forward(self, jobs, machines, allocable_jobs=None, allocable_machines=None, argmax=False):
        job_input_size = jobs.size(0)
        machine_input_size = machines.size(0)

        E_j, E_m = self.get_embedding(jobs, machines)

        g_1 = self.last_j
        j_logits = self.j_att(g_1, E_j)

        ### selecting processes
        if allocable_jobs is not None:
            x = []
            for _ in range(job_input_size):
                if _ not in allocable_jobs:
                    x.append(_)
            if len(x) > 0:
                mask = torch.from_numpy(np.array(x, dtype=int))
                j_logits[mask] = -1e8

        job_softmax = torch.softmax(j_logits, 0)
        job_sampler = Categorical(job_softmax)


        if argmax:
            selected_job = torch.argmax(job_softmax)
        else:
            try:
                selected_job = job_sampler.sample()
            except:
                raise UnboundLocalError;


        as_i = int(selected_job.detach().numpy())

        e_js = E_j[selected_job]
        m_logits = self.m_att(e_js, E_m)
        self.last_j = e_js.detach()
        ### selecting process
        if allocable_machines is not None:
            x = []
            for _ in range(machine_input_size):
                if _ not in allocable_machines[as_i]:
                    x.append(_)
            mask = torch.from_numpy(np.array(x, dtype=int))
            m_logits[mask] = -1e8

        machine_softmax = torch.softmax(m_logits, 0)
        #machine_softmax[machine_softmax < 0] = 0
        machine_sampler = Categorical(machine_softmax)
        if argmax:
            selected_machine = torch.argmax(machine_softmax, -1)
        else:
            selected_machine = machine_sampler.sample()


        logpas = machine_sampler.log_prob(selected_machine) + job_sampler.log_prob(selected_job)
        return (int(selected_job.detach().numpy()), int(selected_machine.detach().numpy())), logpas


class mlp_approximator(nn.Module):
    def __init__(self, job_dim=10, max_job_cnt=100):
        super(mlp_approximator, self).__init__()
        self.job_dim = job_dim
        self.max_job_cnt = max_job_cnt
        self.l1 = nn.Linear(max_job_cnt * job_dim, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 100)
        self.pad = torch.zeros(100, job_dim)

    def reset(self, ):
        pass

    def forward(self, jobs, allocable_jobs=None, argmax=False):
        jobs = jobs[:self.max_job_cnt]
        job_input_size = jobs.size(0)
        if job_input_size < self.max_job_cnt:
            pad = self.pad[:self.max_job_cnt - job_input_size]
            jobs = torch.cat([jobs, pad], 0)
        jobs = jobs.view(-1)
        o = F.relu(self.l1(jobs))
        o = F.relu(self.l2(o))
        j_logits = self.l3(o)
        for _ in range(j_logits.size(0)):
            if _ not in allocable_jobs:
                j_logits[_] = -1e8

        job_softmax = torch.softmax(j_logits, 0)
        job_sampler = Categorical(job_softmax)
        if argmax:
            selected_job = torch.argmax(job_softmax)
        else:
            selected_job = job_sampler.sample()

        return (int(selected_job.detach().numpy()), 0), job_sampler.log_prob(selected_job)