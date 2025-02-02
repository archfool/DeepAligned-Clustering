# -*- coding: UTF-8 -*-
import random

from util import *
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args, tokenizer=None):
        set_seed(args.seed)
        max_seq_lengths = {'clinc': 30, 'stackoverflow': 45, 'banking': 55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        # get all_label_list, generate known_label_list(n_known_cls), cal num_labels.
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        self.train_ori_examples = self.train_labeled_examples + self.train_unlabeled_examples
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train', tokenizer)

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(
            self.train_labeled_examples, self.train_unlabeled_examples, args, tokenizer)
        self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask,
                                                          self.semi_segment_ids, self.semi_label_ids, args)

        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval', tokenizer)
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test', tokenizer)
        self.args = args

    # load data and change into new format
    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []

            # iter for every know_label, random choose a ratio of the corpus tobe labeled, and record their idx
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])
                train_labeled_ids.extend(random.sample(pos, num))

            train_labeled_examples, train_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            return ori_examples

        return examples

    # convert corpus to ids
    def get_semi(self, labeled_examples, unlabeled_examples, args, tokenizer=None):

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length,
                                                        tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length,
                                                          tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    # push corpus into loader
    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size=args.train_batch_size)

        return semi_dataloader

    # convert corpus to ids and push into loader
    def get_loader(self, examples, args, mode='train', tokenizer=None):
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader

    def corpus_dac2cl_train(self, ouput_corpus_path, batch_size, args, pseudo_labels=None, pre_train_file=None):
        if pseudo_labels is None:
            known_label_list = self.known_label_list
            train_labeled_examples = self.train_labeled_examples
        else:
            labels = list(pseudo_labels.cpu().numpy())
            known_label_list = set(labels)
            train_labeled_examples = self.train_ori_examples
            for idx in range(len(self.train_ori_examples)):
                train_labeled_examples[idx].label = labels[idx]

        if True:
            corpus_df = pd.DataFrame(
                [(example.text_a, example.label) for example in train_labeled_examples],
                columns=['text', 'label'])
            corpus_dict = {}
            for label, tmp_corpus_df in corpus_df.groupby(['label']):
                tmp_corpus_lst = []
                for text_a in tmp_corpus_df['text'].to_list():
                    for text_b in tmp_corpus_df['text'].to_list():
                        if text_a != text_b:
                            tmp_corpus_lst.append(text_a + '\t' + text_b)
                random.shuffle(tmp_corpus_lst)
                corpus_dict[label] = tmp_corpus_lst
            corpus = []
            label_candidate = list(corpus_dict.keys())
            label_queue = []
            endless_loop_count = 0
            # count = 0
            while True:
                # if count % 10000 == 0:
                #     print(count, endless_loop_count)
                # count += 1
                label_cur = random.choice(label_candidate)
                if len(corpus_dict[label_cur]) <= 0:
                    label_candidate.remove(label_cur)
                    corpus_dict.pop(label_cur)
                    continue
                elif label_cur in label_queue:
                    if endless_loop_count > batch_size * 10:
                        break
                    else:
                        endless_loop_count += 1
                        continue
                else:
                    corpus_cur = random.choice(corpus_dict[label_cur])
                    corpus.append(corpus_cur)
                    corpus_dict[label_cur].remove(corpus_cur)
                    label_queue.append(label_cur)
                    label_queue = label_queue[-batch_size:]
                    endless_loop_count = 0
        else:
            corpus = []
            # todo rewrite to speed up
            # 遍历所有标签
            for label in known_label_list:
                example_set = []
                # 提取当前标签的所有样本
                for example in train_labeled_examples:
                    if example.label == label:
                        example_set.append(example.text_a)
                if len(example_set) <= 1:
                    continue
                # 将当前标签的所有样本两两组合构成新样本
                for text_a in example_set:
                    for text_b in example_set:
                        if text_a != text_b:
                            corpus.append(text_a + '\t' + text_b)

            # if pre_train_file is not None:
            #     with open(pre_train_file, "r", encoding="utf-8") as f:
            #         pre_corpus = f.read().split("\n")
            #     corpus = corpus + pre_corpus[1:] * 2
            random.shuffle(corpus)

        if pre_train_file is not None:
            corpus = corpus[:int(len(corpus) * self.args.cl_sample_ratio)]
        corpus = ["sent0\tsent1"] + corpus

        if ouput_corpus_path.__contains__("\\"):
            train_file_lst = ouput_corpus_path.split("\\")
            train_file_lst[-1] = "_".join([str(args.model_name), str(args.cl_sample_ratio), train_file_lst[-1]])
            train_file = "\\".join(train_file_lst)
        elif ouput_corpus_path.__contains__("/"):
            train_file_lst = ouput_corpus_path.split("/")
            train_file_lst[-1] = "_".join([str(args.model_name), str(args.cl_sample_ratio), train_file_lst[-1]])
            train_file = "/".join(train_file_lst)
        else:
            train_file = "_".join([str(args.model_name), str(args.cl_sample_ratio), ouput_corpus_path])

        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(corpus))

        # def corpus_dac2cl_eval(self, ouput_corpus_path, test_or_eval="eval"):
        #     if "eval" == test_or_eval:
        #         examples = self.eval_examples
        #     elif "test" == test_or_eval:
        #         examples = self.test_examples
        #     else:
        #         example = None
        #         print("Input Para test_or_eval ERROR!")
        #
        #     example_set = [example.text_a for example in examples]
        #
        #     return example_set
        return train_file


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    # corpus should be saved into 3 files, and named by:train, dev, test
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


# todo havn't read
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
