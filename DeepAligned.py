from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
import time
import transformers
import simcse_train
import CL_fit_DAC
import pretrain_manage_dac_cl
import logging
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)


class ModelManager:

    def __init__(self, args, data, pretrained_model=None):

        # todo 加载模型
        if pretrained_model is None:
            print("load model from a saved bin file.")
            pretrained_model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                print("find args.pretrain_dir bin file, ignore args.bert_model bin file.")
                pretrained_model = self.restore_model(args.pretrained_model)
        else:
            print("load model direct from previous train task.")
        self.pretrained_model = pretrained_model

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # todo tmp add
        self.pretrained_model.to(self.device)

        # todo 当cluster_num_factor<=1时，不会执行低质簇剔除的策略
        if args.cluster_num_factor > 1:
            # todo 进行聚类，剔除低密度的簇，统计符合条件的簇的个数
            # todo 只会在训练deepcluster之前进行一次簇个数的估计
            self.num_labels = self.predict_k(args, data)
        else:
            self.num_labels = data.num_labels

        # todo 根据估计出的簇个数，调整label个数，重置预训练模型
        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=self.num_labels)

        # todo 加载self.pretrained_model的模型参数至self.model，除了分类层
        if args.pretrain:
            self.load_pretrained_model(args)

        # todo 冻结除了12层和pooler层之外的所有参数
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        # todo tmp del
        self.model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_features_labels(self, dataloader, model, args):

        # todo debug add change device
        # model.to(self.device)
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        # for batch in tqdm(dataloader, desc="Extracting representation"):
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        print("{}\tBegin KMeans".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        print("{}\tEnd KMeans".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        # todo 簇的筛选门限值
        drop_out = len(feats) / data.num_labels
        print('drop threshold:', drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num', num_labels)

        return num_labels

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # todo 不知道weight_decay是干吗用的
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def evaluation(self, args, data):

        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results', results)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])

        cm = confusion_matrix(y_true, y_pred)
        print('confusion matrix\n', cm)
        self.test_results = results

        self.save_results(args)

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_

            DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
            # linear_sum_assignment函数输入为cost矩阵。cost[i, j]表示工人i执行任务j所要花费的代价。
            # linear_sum_assignment函数输出row_ind和col_ind。row_ind表示选择哪几个工人，col_ind表示工人做哪个工作。
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels, args.feat_dim).to(self.device)

            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)
            pseudo_labels = km.labels_

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        return train_dataloader

    def train(self, args, data):

        best_score = 0
        best_model = None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            print("{}\tEpoch:\t{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch))
            feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            print("\n{}\tBegin KMeans".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            km = KMeans(n_clusters=self.num_labels).fit(feats)
            print("{}\tEnd KMeans".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            score = metrics.silhouette_score(feats, km.labels_)
            print('score(the bigger the better)', score)

            # todo 判定early_stop
            if score > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model = best_model
                    break

            # todo 生成伪标签，更新旧标签为新伪标签
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            # for batch in tqdm(train_dataloader, desc="Pseudo-Training"):
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            tr_loss = tr_loss / nb_tr_steps
            print('train_loss', tr_loss)
            if epoch % 100 == 0:
                print("=============Begin in_batch Eval=============")
                self.evaluation(args, data)
                print("=============End in_batch Eval=============")

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed,
               self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results' + '.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    env_paras = {
        'DATASET': 'clinc',
        'MODEL_NAME': 'bert-base-uncased',
    }
    # todo step_1 读取配置参数
    parser = init_model()
    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    if args.use_CL:
        parser_CL = transformers.HfArgumentParser(
            (simcse_train.ModelArguments,
             simcse_train.DataTrainingArguments,
             simcse_train.OurTrainingArguments)
            , allow_abbrev=False)
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser_CL.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args, remaining_args = parser_CL.parse_args_into_dataclasses(
                return_remaining_strings=True)

    if os.path.exists("D:"):
        # args.bert_model = r'E:\data\huggingface\bert-base-uncased'
        # args.bert_model = r'E:\data\huggingface\unsup-simcse-bert-base-uncased'
        args.bert_model = r'E:\data\my-unsup-simcse-bert-base-uncased'
        args.dataset = 'clinc'
        # args.max_seq_length = 128
        args.num_train_epochs_dac = 2
        args.labeled_ratio = 0.4
        args.seed_dac = 1234
    elif os.path.exists("/media/archfool/"):
        args.bert_model = r'/media/archfool/data/data/huggingface/unsup-simcse-bert-base-uncased'
        # args.bert_model = r'/media/archfool/data/data/huggingface/bert-base-uncased'
    # else:
    #     args.bert_model = 'undefined'

    if args.use_CL:
        if (
                os.path.exists(training_args.output_dir)
                and os.listdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
        )

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed before initializing model.
        set_seed(training_args.seed)

    # todo step_2 读取数据
    data = Data(args)

    # todo step_4 加载模型
    if args.pretrain:
        print('Pre-training begin...')
        if args.use_CL:
            manager_p = pretrain_manage_dac_cl.PretrainModelManagerCL(args, data)
            if model_args.model_name_or_path:
                if 'roberta' in model_args.model_name_or_path:
                    model = RobertaForCL.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        model_args=model_args
                    )
                elif 'bert' in model_args.model_name_or_path:
                    model = BertForCL.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        model_args=model_args
                    )
                    if model_args.do_mlm:
                        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                        model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
                logger.info("Training new model from scratch")
                model = AutoModelForMaskedLM.from_config(config)

            model.resize_token_embeddings(len(tokenizer))
        else:
            manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data)

    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

    manager.save_results(args)
