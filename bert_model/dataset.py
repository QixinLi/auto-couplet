# coding=utf-8
import random
from copy import deepcopy
from typing import List
import datasets
from dataclasses import dataclass
from bert_model.utils import *

@dataclass
class MLMConfig(datasets.BuilderConfig):
    langs: List[str] = None
    modes: List[str] = None
    seed: int = 10000
    pad_idx: int = -100
    data_dir: str = None
    max_seq_len: int = 128
    debug: bool = False


class CSMlmPlusDataset(datasets.GeneratorBasedBuilder):
    config: MLMConfig
    BUILDER_CONFIG_CLASS = MLMConfig

    def __init__(self,tokenizer, **config_kwargs):
        super().__init__(**config_kwargs)
        random.seed(self.config.seed)
        self.tokenizer = tokenizer

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int64")),
                "attention_mask": datasets.Sequence(datasets.Value("int64")),
                "token_type_ids": datasets.Sequence(datasets.Value("int64")),
                "labels": datasets.Sequence(datasets.Value("int64"))
            })
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN)
        ]

    def _generate_examples(self):
        line_num = 0
        handler = data_handler(
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_len
        )
        handler.convert_examples_to_features()

        for utterance in handler.utterances:
            if line_num > 100 and self.config.debug:
                break
            if not (utterance.text_a and utterance.text_b):
                continue
            item = handler.convert_one_utterance_to_feature_v2(utterance)
            # print(item)
            yield line_num, {
                "input_ids": item.input_ids,
                "attention_mask": item.input_mask,
                "token_type_ids": item.segment_ids,
                "labels": item.label_ids,
                "domain_labels": item.domain_id
            }
            line_num += 1