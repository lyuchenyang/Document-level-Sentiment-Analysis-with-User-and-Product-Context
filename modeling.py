import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers import BertPreTrainedModel, BertModel, InputExample, RobertaModel, BertForSequenceClassification, RobertaForSequenceClassification, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import RobertaClassificationHead
from transformers.modeling_bert import BertOnlyMLMHead


class SecondPretrainedBert(BertPreTrainedModel):
    def __init__(self, config, num_embeddings):
        super().__init__(config)

        self.config=config
        self.bert = BertForSequenceClassification(config)
        self.embedding_matrix = nn.Embedding(num_embeddings, config.hidden_size)

    def forward(self, inputs, user_product):
        outputs = self.bert(**inputs)

        loss, last_hidden_states = outputs[0], outputs[2][self.config.num_hidden_layers-1]

        all_cls = last_hidden_states[:, 0, :]

        _all_cls = torch.cat([all_cls.detach().clone(), all_cls.detach().clone()], dim=0)
        up_embeddings = self.embedding_matrix(user_product)

        self.embedding_matrix.weight.index_copy(0, user_product.view(-1),
                                                up_embeddings.view(-1, self.config.hidden_size).detach() + _all_cls)

        return loss


class IncrementalContextBert(BertPreTrainedModel):

    def __init__(self, config, num_embeddings, up_vocab):
        super().__init__(config)

        self.bert = BertModel(config)

        if config.do_shrink:
            self.embedding = nn.Embedding(num_embeddings, config.inner_size)
            self.to_hidden_size = nn.Linear(config.inner_size, config.hidden_size)
            self.to_inner_size = nn.Linear(config.hidden_size, config.inner_size)
        else:
            self.embedding = nn.Embedding(num_embeddings, config.hidden_size)

        self.multi_head_attention = torch.nn.MultiheadAttention(config.hidden_size, config.attention_heads)

        # Linear layers used to transform cls token, user and product embeddings
        self.linear_t = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_u = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_p = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_update = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_f = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_g = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)

        # An empirical initializad number, still needed to be explored
        self.alpha = nn.Parameter(torch.tensor(-10, dtype=torch.float), requires_grad=True)

        self.up_vocab = up_vocab

        self.init_weights()

    def forward(self, inputs, user_product, up_indices=None, up_embeddings=None):
        if up_indices is not None and up_embeddings is not None:
            p_up_embeddings = self.embedding(up_indices)
            update_embeddings = p_up_embeddings + self.sigmoid(self.alpha)*up_embeddings
            with torch.no_grad():
                self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.bert(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0].transpose(0, 1), outputs[1]

        up_embeddings = self.embedding(user_product)

        if self.config.do_shrink:
            up_embeddings = self.to_hidden_size(up_embeddings)

        att_up = self.multi_head_attention(up_embeddings.transpose(0, 1), last_hidden_states, last_hidden_states)
        att_u, att_p = att_up[0][0, :, :], att_up[0][1, :, :]

        z_cls = self.sigmoid(self.linear_t(cls_hidden_states))
        z_att_u, z_att_p = self.sigmoid(self.linear_u(att_u)), self.sigmoid(self.linear_p(att_p))

        z_u = self.sigmoid(z_cls + z_att_u)
        z_p = self.sigmoid(z_cls + z_att_p)

        cls_input = cls_hidden_states + z_u * att_u + z_p * att_p

        logits = self.classifier(cls_input)
        # logits = self.softmax(logits)

        new_up_embeddings = torch.cat([z_att_u, z_att_p], dim=0)

        if self.config.do_shrink:
            new_up_embeddings = self.to_inner_size(new_up_embeddings)

        return logits, user_product.view(-1).detach(), new_up_embeddings


class IncrementalContextRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, num_embeddings, up_vocab):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        if config.do_shrink:
            self.embedding = nn.Embedding(num_embeddings, config.inner_size)
            self.to_hidden_size = nn.Linear(config.inner_size, config.hidden_size)
            self.to_inner_size = nn.Linear(config.hidden_size, config.inner_size)
        else:
            self.embedding = nn.Embedding(num_embeddings, config.hidden_size)

        self.multi_head_attention = torch.nn.MultiheadAttention(config.hidden_size, config.attention_heads)

        # Linear layers used to transform cls token, user and product embeddings
        self.linear_t = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_u = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_p = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_update = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_f = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_g = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)

        # An empirical initializad number, still needed to be explored
        self.alpha = nn.Parameter(torch.tensor(-10, dtype=torch.float), requires_grad=True)

        self.up_vocab = up_vocab

        self.init_weights()

    def forward(self, inputs, user_product, up_indices=None, up_embeddings=None):
        if up_indices is not None and up_embeddings is not None:
            p_up_embeddings = self.embedding(up_indices)
            update_embeddings = p_up_embeddings + self.sigmoid(self.alpha) * up_embeddings
            with torch.no_grad():
                self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.roberta(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0].transpose(0, 1), outputs[1]

        up_embeddings = self.embedding(user_product)

        if self.config.do_shrink:
            up_embeddings = self.to_hidden_size(up_embeddings)

        att_up = self.multi_head_attention(up_embeddings.transpose(0, 1), last_hidden_states, last_hidden_states)
        att_u, att_p = att_up[0][0, :, :], att_up[0][1, :, :]

        z_cls = self.sigmoid(self.linear_t(cls_hidden_states))
        z_att_u, z_att_p = self.sigmoid(self.linear_u(att_u)), self.sigmoid(self.linear_p(att_p))

        z_u = self.sigmoid(z_cls + z_att_u)
        z_p = self.sigmoid(z_cls + z_att_p)

        cls_input = cls_hidden_states + z_u * att_u + z_p * att_p

        logits = self.classifier(cls_input)
        # logits = self.softmax(logits)

        new_up_embeddings = torch.cat([z_att_u, z_att_p], dim=0)

        if self.config.do_shrink:
            new_up_embeddings = self.to_inner_size(new_up_embeddings)

        return logits, user_product.view(-1).detach(), new_up_embeddings


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
