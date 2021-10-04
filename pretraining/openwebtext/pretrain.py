import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import logging
import random
from dataclasses import dataclass
from time import time

import numpy as np
import torch
from electra_pytorch import Electra
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from openwebtext import arg
from openwebtext.dataset import load_owt, new_tokenizer, wrap_example_builder

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, DebertaConfig, DebertaModel,
                          DebertaPreTrainedModel)
from transformers.activations import get_activation,ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import MaskedLMOutput

########################################################################################################
## args

@dataclass
class Args:
    data_dir: arg.Str = 'data/openwebtext_features'
    data_vocab_file: arg.Str = 'data/vocab.txt'
    data_n_tensors_per_file: arg.Int = 2048
    data_max_seq_length: arg.Int = 128

    gpu: arg.Int = 0
    gpu_enabled: arg.Bool = True
    gpu_deterministic: arg.Bool = False
    gpu_mixed_precision: arg.Bool = True
    distributed_port: arg.Int = 8888
    distributed_enabled: arg.Bool = True
    distributed_world_size: arg.Int = 4

    model_generator: arg.Str = 'pretraining/openwebtext/small_generator.json'
    model_discriminator: arg.Str = 'pretraining/openwebtext/small_discriminator.json'
    model_mask_prob: arg.Float = 0.15

    opt_lr: arg.Float = 5e-4
    opt_batch_size: arg.Int = 128 // (distributed_world_size if distributed_enabled else 1)
    opt_warmup_steps: arg.Int = 10_000
    opt_num_training_steps: arg.Int = 100_0000

    step_log: arg.Int = 100
    step_ckpt: arg.Int = 10_000


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits

@dataclass
class ElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTraining`.
    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DebertaForPreTraining(DebertaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.electra = DebertaModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        Returns:
        Examples::
            >>> from transformers import ElectraTokenizer, ElectraForPreTraining
            >>> import torch
            >>> tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            >>> model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> logits = model(input_ids).logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


########################################################################################################
## train

def train(rank, args):

    #######################
    ## distributed

    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    if args.gpu_enabled:
        device = torch.device('cuda:{}'.format(rank))
    else:
        device = torch.device('cpu')

    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0


    #######################
    ## preamble

    set_gpus(rank)
    set_seed(rank)
    # set_cuda(deterministic=args.gpu_deterministic)

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=False)

    setup_logging(filename=f'{output_dir}/output.log', console=is_master)


    #######################
    ## dataset

    tokenizer = new_tokenizer(vocab_file=args.data_vocab_file)
    vocab_size = len(tokenizer.vocab)
    ds_train = wrap_example_builder(dataset=load_owt(owt_dir=args.data_dir, n_tensors_per_file=args.data_n_tensors_per_file), vocab=tokenizer.vocab, max_length=args.data_max_seq_length)

    pad_token_id = tokenizer.vocab['[PAD]']
    mask_token_id = tokenizer.vocab['[MASK]']
    cls_token_id = tokenizer.vocab['[CLS]']
    sep_token_id = tokenizer.vocab['[SEP]']

    assert pad_token_id == 0
    assert cls_token_id == 101
    assert sep_token_id == 102
    assert mask_token_id == 103

    def collate_batch(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        input_mask = torch.nn.utils.rnn.pad_sequence([example['input_mask'] for example in examples], batch_first=True, padding_value=pad_token_id)
        segment_ids = torch.nn.utils.rnn.pad_sequence([example['segment_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        return input_ids, input_mask, segment_ids

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    ds_train_loader = iter(cycle(DataLoader(ds_train, batch_size=args.opt_batch_size, collate_fn=collate_batch)))


    #######################
    ## model

    def to_distributed_model(model):
        return model if not args.distributed_enabled else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    def tie_weights(generator, discriminator):
        generator.deberta.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.deberta.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.deberta.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings

    class LogitsAdapter(torch.nn.Module):
        def __init__(self, adaptee):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]

    

    generator = DebertaForMaskedLM(AutoConfig.from_pretrained(args.model_generator))
    discriminator = DebertaForPreTraining(DebertaConfig.from_pretrained(args.model_discriminator))

    tie_weights(generator, discriminator)

    model = to_distributed_model(Electra(
        LogitsAdapter(generator),
        LogitsAdapter(discriminator),
        num_tokens = vocab_size,
        mask_token_id = mask_token_id,
        pad_token_id = pad_token_id,
        mask_prob = args.model_mask_prob,
        mask_ignore_token_ids = [tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
        random_token_prob = 0.0).to(device))


    #######################
    ## optimizer

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)


    #######################
    ## train

    t, steps_s, eta_m = time(), 0., 0

    for step in range(args.opt_num_training_steps+1):
        input_ids, input_mask, segment_ids = next(ds_train_loader)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        assert input_ids.shape[1] <= args.data_max_seq_length

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

        if step > 0 and step % 100 == 0:
            t2 = time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        if step % 200 == 0:
            logger.info(np.array2string(disc_labels[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(np.array2string(disc_pred[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))

        if step > 0 and step % args.step_ckpt == 0 and is_master:
            discriminator.electra.save_pretrained(f'{args.output_dir}/ckpt/{step}')

########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(filename, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


########################################################################################################
## main

def main():

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/ckpt', exist_ok=False)
    copy_source(__file__, output_dir)

    # args
    args = arg.parse_to(Args)
    args.output_dir = output_dir
    args.exp_id = exp_id

    # distributed
    if args.distributed_enabled:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        torch.multiprocessing.spawn(train, nprocs=args.distributed_world_size, args=(args,))
    else:
        train(rank=args.gpu, args=args)


if __name__ == '__main__':
    main()
