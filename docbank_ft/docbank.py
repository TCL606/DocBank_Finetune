import os
import random
from itertools import chain
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import logging
from layoutlmft.data.data_args import DocBankDataTrainingArguments
import json
from layoutlmft.data.utils import normalize_bbox
from torch.nn import CrossEntropyLoss
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    LayoutLMConfig,
    LayoutLMForTokenClassification
)

logger = logging.getLogger('__name__')

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizer),
}

class DocBankMetaData:
    def __init__(self, filepath, pagesize, words, bboxes, rgbs, fontnames, structures, img):
        assert len(words) == len(bboxes)
        assert len(bboxes) == len(rgbs)
        assert len(rgbs) == len(fontnames)
        assert len(fontnames) == len(structures)
        
        self.filepath = filepath
        self.pagesize = pagesize
        self.words = words
        self.bboxes = bboxes
        self.rgbs = rgbs
        self.fontnames = fontnames
        self.structures = structures
        self._infos = None
        self.img = img

    @classmethod
    def label2id(cls, label) -> int:
        if label == 'paragraph':
            return 0
        elif label == 'title':
            return 1
        elif label == 'equation':
            return 2
        elif label == 'reference':
            return 3
        elif label == 'section':
            return 4
        elif label == 'list':
            return 5
        elif label == 'table':
            return 6
        elif label == 'caption':
            return 7
        elif label == 'author':
            return 8
        elif label == 'abstract':
            return 9
        elif label == 'footer':
            return 10
        elif label == 'date':
            return 11
        elif label == 'figure':
            return 12
        else:
            raise Exception('Invalid label!')       

class DocBankDataset(Dataset):
    def __init__(self, args: DocBankDataTrainingArguments, tokenizer, mode: str):
        self.img_dir = args.img_dir
        self.txt_dir = args.txt_dir
        self.json_dir = args.json_dir
        self.tokenizer = tokenizer
        self.mode = mode
        self.args = args
        self.pad_token_label_id=CrossEntropyLoss().ignore_index

        with open(os.path.join(self.json_dir, f'500K_{mode}.json'), 'r') as fp:
            data = json.load(fp)

        self.basenames_list = [metadata['file_name'].replace('_ori.jpg', '') for metadata in data['images']]
        # if mode != 'train':
        #     self.basenames_list = self.basenames_list[: 100]
                
    def __getitem__(self, index):
        example = self.read_example_from_file(index=index)
        feature = self.convert_example_to_feature(
            example,
            self.args.max_seq_length,
            cls_token_at_end=bool(self.args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.args.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(self.args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.args.model_type in ["xlnet"] else 0,
            pad_token_label_id=self.pad_token_label_id
        )
        return feature

    def __len__(self):
        return len(self.basenames_list)

    def read_example_from_file(self, index):
        basename = self.basenames_list[index]        
        txt_file = basename + '.txt'
        img_file = basename + '_ori.jpg'
        
        words = []
        bboxes = []
        rgbs = []
        fontnames = []
        structures = []

        im = Image.open(os.path.join(self.img_dir, img_file))
        pagesize = im.size

        with open(os.path.join(self.txt_dir, txt_file), 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                tts = line.split('\t')
                if not len(tts) == 10:
                    logger.warning('Incomplete line in file {}'.format(txt_file))
                    continue
                
                word = tts[0]
                bbox = list(map(int, tts[1:5]))
                rgb = list(map(int, tts[5:8]))
                fontname = tts[8]
                structure = tts[9].replace('\n', '')
                
                words.append(word)
                bboxes.append(bbox)
                rgbs.append(rgb)
                fontnames.append(fontname)
                structures.append(DocBankMetaData.label2id(structure))
        
        example = DocBankMetaData(
            filepath = os.path.join(self.txt_dir, txt_file),
            pagesize = pagesize,
            words = words,
            bboxes = bboxes,
            rgbs = rgbs,
            fontnames = fontnames,
            structures = structures,
            img = im
        )
        return example

    def convert_example_to_feature(
        self,
        example: DocBankMetaData,
        max_seq_length,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # filepath = example.filepath
        pagesize = example.pagesize
        # width, height = pagesize

        tokens = []
        token_boxes = []
        # actual_bboxes = []
        label_ids = []
        for word, label, box in zip(
            example.words, example.structures, example.bboxes #, example.actual_bboxes
        ):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            # actual_bboxes.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label] + [pad_token_label_id] * (len(word_tokens) - 1) if len(word_tokens) > 0 else []
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # special_tokens_count = 3 if sep_token_extra else 2
        tokens = [tokens[i * max_seq_length: min((i + 1) * max_seq_length, len(tokens))] for i in range(len(tokens) // max_seq_length + 1)]
        token_boxes = [token_boxes[i * max_seq_length: min((i + 1) * max_seq_length, len(token_boxes))] for i in range(len(token_boxes) // max_seq_length + 1)]
        # actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
        label_ids = [label_ids[i * max_seq_length: min((i + 1) * max_seq_length, len(label_ids))] for i in range(len(label_ids) // max_seq_length + 1)]

        input_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [[1 if mask_padding_with_zero else 0] * len(i) for i in input_ids]

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids[-1])
        if pad_on_left:
            input_ids[-1] = ([pad_token] * padding_length) + input_ids[-1]
            input_mask[-1] = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask[-1]
            # segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids[-1] = ([pad_token_label_id] * padding_length) + label_ids[-1]
            token_boxes[-1] = ([pad_token_box] * padding_length) + token_boxes[-1]
        else:
            input_ids[-1] += [pad_token] * padding_length
            input_mask[-1] += [0 if mask_padding_with_zero else 1] * padding_length
            # segment_ids += [pad_token_segment_id] * padding_length
            label_ids[-1] += [pad_token_label_id] * padding_length
            token_boxes[-1] += [pad_token_box] * padding_length

        if len(input_ids) > 10:
            input_ids = input_ids[: 10]
            input_mask = input_ids[: 10]
            label_ids = label_ids[: 10]
            token_boxes = token_boxes[: 10]

        # input_ids = input_ids[: 1]
        # input_mask = input_ids[: 1]
        # label_ids = label_ids[: 1]
        # token_boxes = token_boxes[: 1]

        assert len(input_ids[-1]) == max_seq_length
        assert len(input_mask[-1]) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        assert len(label_ids[-1]) == max_seq_length
        assert len(token_boxes[-1]) == max_seq_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'label_ids': label_ids,
            'bbox': token_boxes,
            # 'file': filepath
            # "image": example.img
        }

@dataclass
class DocBankCollator:
    def __call__(self, features):   
        batch = dict()
        batch['labels'] = torch.tensor(list(chain(*[feature['label_ids'] for feature in features])), dtype=torch.int64)
        batch['bbox'] = torch.tensor(list(chain(*[feature['bbox'] for feature in features])), dtype=torch.int32)
        batch['input_ids'] = torch.tensor(list(chain(*[feature['input_ids'] for feature in features])), dtype=torch.int32)
        batch['attention_mask'] = torch.tensor(list(chain(*[feature['attention_mask'] for feature in features])), dtype=torch.int32)
        # batch['token_type_ids'] = torch.zeros(batch['labels'].shape, dtype=torch.int64)
        # batch['image'] = torch.tensor([feature['image'] for feature in features])
        return batch