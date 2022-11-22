from dataclasses import dataclass, field
from typing import Optional
from layoutlmft.data.data_args import DataTrainingArguments

@dataclass
class DocBankDataTrainingArguments(DataTrainingArguments):
    txt_dir: str = field(
        default='/root/pubdatasets/DocBank/txt/DocBank_500K_txt', metadata={"help": "Docbank txt dir"}
    )
    img_dir: str = field(
        default='/root/pubdatasets/DocBank/img/DocBank_500K_ori_img', metadata={"help": "Docbank img dir"}
    )
    json_dir: str = field(
        default='/root/pubdatasets/DocBank/coco', metadata={"help": "Docbank json file dir"}
    )
    max_seq_length: int = field(
        default=512, metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    model_type: str = field(
        default='layoutlm', metadata={
            "help": "model type"
        }
    )
    require_image: bool = field(
        default=False, metadata={
            'help': 'whether to use img as input'
        }
    )