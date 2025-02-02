# -*- coding: UTF-8 -*-

from util import *


class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        # todo 加载模型，模型参数是随机初始化的
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, centroids=None,
                labeled=False, feature_ext=False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                    output_all_encoded_layers=False)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        # todo embd representation后只接了一层
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits
