from itertools import chain

import model.module as modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel


class GeNetModel(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024):
        super(GeNetModel, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h, num_filters, resnet_out)
        self.logits = modules.GeNetLogitLayer2(resnet_out, list_num_classes)

        self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        x = self.preresnet(x)

        x = self.gresnet(x)

        x = self.logits(x)
        return x


class GeNetModelDeepSet(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 deepset_hidden=2048,
                 deepset_out=1024,
                 extra_phi_layer=True,
                 deepset_activation='relu',
                 deepset_dropout=0.5,
                 reset_weights=False,
                 logit_layer_type='type2',
                 skip='none',
                 bn_running_stats=True,
                 log_output=False,
                 resnet_checkpoint=None):
        super(GeNetModelDeepSet, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h,
                                           num_filters,
                                           resnet_out,
                                           bn_running_stats=bn_running_stats)

        assert skip in ['none', 'connection', 'completely']
        self.skip = skip

        # \rho(\sum(\phi(x)))
        if self.skip == 'completely':
            assert extra_phi_layer == False
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           average=True,
                                           rho=False)
            logit_input_size = resnet_out
        else:
            if extra_phi_layer:
                phi = nn.Sequential(nn.Linear(resnet_out, resnet_out),
                                    nn.ReLU(inplace=True))
            else:
                phi = None
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           phi=phi,
                                           average=True)
            logit_input_size = deepset_out
            if self.skip == 'connection':
                assert logit_input_size == resnet_out

        assert logit_layer_type in ['type1', 'type2']

        if logit_layer_type == 'type1':
            self.logits = modules.GeNetLogitLayer(logit_input_size,
                                                  list_num_classes)
        elif logit_layer_type == 'type2':
            self.logits = modules.GeNetLogitLayer2(logit_input_size,
                                                   list_num_classes)

        if log_output:
            self.prob = nn.LogSoftmax(dim=1)
        else:
            self.prob = nn.Softmax(dim=1)

        if reset_weights:
            self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

        if resnet_checkpoint:
            checkpoint = torch.load(str(resnet_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('preresnet') or k.startswith('gresnet'):
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.preresnet.requires_grad_(False)
            self.gresnet.requires_grad_(False)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.preresnet(x)

        x = self.gresnet(x)
        resout = x  # [batch*bag, rest]

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)  # [batch_size, rest]

        if self.skip == 'connection':
            resout = resout.view(batch_size, bag_size, *resout.size()[1:])
            resout = torch.mean(resout, dim=1)
            x = x + resout

        x = self.logits(x)
        # print(x)
        x = [self.prob(j) for j in x]
        # print(x)
        return x


class GeNetModelMILAttention(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 reset_weights=False,
                 logit_layer_type='type2',
                 bn_running_stats=True,
                 pool_hidden=128,
                 pool_n_attentions=30,
                 pool_gate=False,
                 resnet_checkpoint=None):
        super(GeNetModelMILAttention, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h,
                                           num_filters,
                                           resnet_out,
                                           bn_running_stats=bn_running_stats)

        self.pool = modules.MILAttentionPool(resnet_out,
                                             pool_hidden,
                                             pool_n_attentions,
                                             gated=pool_gate)
        logit_input_size = resnet_out * pool_n_attentions

        self.logits = modules.GeNetLogitLayer2(logit_input_size,
                                               list_num_classes)

        self.prob = nn.LogSoftmax(dim=1)

        if reset_weights:
            self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

        if resnet_checkpoint:
            checkpoint = torch.load(str(resnet_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('preresnet') or k.startswith('gresnet'):
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.preresnet.requires_grad_(False)
            self.gresnet.requires_grad_(False)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.preresnet(x)

        x = self.gresnet(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)  # [batch_size, rest]

        x = self.logits(x)
        # print(x)
        x = [self.prob(j) for j in x]
        # print(x)
        return x


class GeNetModelDeepSetCounting(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 deepset_hidden=2048,
                 deepset_out=1024,
                 extra_phi_layer=True,
                 deepset_activation='relu',
                 deepset_dropout=0.5,
                 reset_weights=False,
                 logit_layer_type='type2',
                 skip='none',
                 bn_running_stats=True):
        super().__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h,
                                           num_filters,
                                           resnet_out,
                                           bn_running_stats=bn_running_stats)

        assert skip in ['none', 'connection', 'completely']
        self.skip = skip

        # \rho(\sum(\phi(x)))
        if self.skip == 'completely':
            assert extra_phi_layer == False
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           average=False,
                                           rho=False)
            logit_input_size = resnet_out
        else:
            if extra_phi_layer:
                phi = nn.Sequential(nn.Linear(resnet_out, resnet_out),
                                    nn.ReLU(inplace=True))
            else:
                phi = None
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           phi=phi,
                                           average=False)
            logit_input_size = deepset_out
            if self.skip == 'connection':
                assert logit_input_size == resnet_out

        assert logit_layer_type in ['type1', 'type2']

        if logit_layer_type == 'type1':
            self.logits = modules.GeNetLogitLayer(logit_input_size,
                                                  list_num_classes)
        elif logit_layer_type == 'type2':
            self.logits = modules.GeNetLogitLayer2(logit_input_size,
                                                   list_num_classes)

        if reset_weights:
            self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.preresnet(x)

        x = self.gresnet(x)
        resout = x

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)

        if self.skip == 'connection':
            resout = resout.view(batch_size, bag_size, *resout.size()[1:])
            resout = torch.sum(resout, dim=1)
            x = x + resout

        x = self.logits(x)
        return x


class DeepMicrobes(BaseModel):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 lstm_dim,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 row,
                 da,
                 keep_prob,
                 sparse_gradient=True):
        super().__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.lstm_dim = lstm_dim
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.row = row
        self.da = da
        self.keep_prob = keep_prob

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)
        self.bilstm = modules.BidirectionalLSTM(self.embedding_dim,
                                                self.lstm_dim)
        self.attention = modules.AttentionLayer(2 * self.lstm_dim, self.da,
                                                self.row)
        self.fc1 = nn.Linear(self.row * 2 * self.lstm_dim, self.mlp_dim)
        self.drop1 = nn.Dropout(p=1 - self.keep_prob)

        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.drop2 = nn.Dropout(p=1 - self.keep_prob)

        self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)

        self.reset_parameters()
        # check_parameter_number(self, True, True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        self.bilstm.reset_parameters()
        self.attention.reset_parameters()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding':
                self.embedding.parameters(recurse=recurse),
                'bilstm':
                self.bilstm.parameters(recurse=recurse),
                'attention':
                self.attention.parameters(recurse=recurse),
                'fc':
                chain(self.fc1.parameters(recurse=recurse),
                      self.drop1.parameters(recurse=recurse),
                      self.fc2.parameters(recurse=recurse),
                      self.drop2.parameters(recurse=recurse),
                      self.fc3.parameters(recurse=recurse))
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.attention(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        x = self.drop2(x)

        logits = self.fc3(x)
        return logits


class EmbedPool(BaseModel):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True):
        super().__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)

        self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding': self.embedding.parameters(recurse=recurse),
                'fc1': self.fc1.parameters(recurse=recurse),
                'fc2': self.fc2.parameters(recurse=recurse),
                'fc3': self.fc3.parameters(recurse=recurse)
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)

        logits = self.fc3(x)
        return logits


class EmbedPoolDS(EmbedPool):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True,
                 ds_dropout=0.0,
                 ds_position='after_embedding'):
        super().__init__(list_num_classes, all_levels, selected_level, mlp_dim,
                         vocab_size, embedding_dim, sparse_gradient)
        self.deepset = modules.DeepSet(2 * self.embedding_dim,
                                       0,
                                       0,
                                       phi=None,
                                       activation='relu',
                                       dropout=ds_dropout,
                                       average=True,
                                       rho=False)
        self.prob = nn.LogSoftmax(dim=1)

        assert ds_position in [
            'after_embedding', 'before_logits', 'after_logits'
        ]
        ds_position_dict = {
            'after_embedding': self.forward_ae,
            'before_logits': self.forward_bl,
            'after_logits': self.forward_al
        }
        self.forward = ds_position_dict[ds_position]

    def embedding_forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)
        return x

    def fc_forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        return x

    def forward_ae(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)

        x = self.fc_forward(x)
        logits = self.fc3(x)
        return self.prob(logits)

    def forward_bl(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)

        logits = self.fc3(x)
        return self.prob(logits)

    def forward_al(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)
        x = self.fc3(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        logits = self.deepset(x)

        return self.prob(logits)


class EmbedPoolMILAttention(EmbedPool):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True,
                 pool_position='after_embedding',
                 pool_hidden=128,
                 pool_n_attentions=30,
                 pool_gate=False,
                 embedding_checkpoint=None):
        super(EmbedPool, self).__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)

        assert pool_position in [
            'after_embedding', 'before_logits', 'after_logits'
        ]

        if pool_position == 'after_embedding':
            self.pool = modules.MILAttentionPool(2 * self.embedding_dim,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)
            self.fc1 = nn.Linear(2 * self.embedding_dim * pool_n_attentions,
                                 self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)
        elif pool_position == 'before_logits':
            self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.pool = modules.MILAttentionPool(self.mlp_dim,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)
            self.fc3 = nn.Linear(self.mlp_dim * pool_n_attentions,
                                 self.num_classes)
        elif pool_position == 'after_logits':
            assert pool_n_attentions == 1
            self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)
            self.pool = modules.MILAttentionPool(self.num_classes,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)

        self.prob = nn.LogSoftmax(dim=1)
        self.reset_parameters()

        pool_position_dict = {
            'after_embedding': self.forward_ae,
            'before_logits': self.forward_bl,
            'after_logits': self.forward_al
        }
        self.forward = pool_position_dict[pool_position]
        if embedding_checkpoint:
            checkpoint = torch.load(str(embedding_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('embedding'):
                    print('Loading embedding...')
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.embedding.requires_grad_(False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        nn.init.xavier_uniform_(self.pool.fc1.weight)
        nn.init.zeros_(self.pool.fc1.bias)

        nn.init.xavier_uniform_(self.pool.fc2.weight)
        nn.init.zeros_(self.pool.fc2.bias)

        if self.pool.fc3:
            nn.init.xavier_uniform_(self.pool.fc3.weight)
            nn.init.zeros_(self.pool.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding': self.embedding.parameters(recurse=recurse),
                'pool': self.pool.parameters(recurse=recurse),
                'fc1': self.fc1.parameters(recurse=recurse),
                'fc2': self.fc2.parameters(recurse=recurse),
                'fc3': self.fc3.parameters(recurse=recurse)
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def embedding_forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)
        return x

    def fc_forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)

        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        return x

    def forward_ae(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)

        x = self.fc_forward(x)
        logits = self.fc3(x)
        return self.prob(logits)

    def forward_bl(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)

        logits = self.fc3(x)
        return self.prob(logits)

    def forward_al(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)
        x = self.fc3(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        logits = self.pool(x)

        return self.prob(logits)
