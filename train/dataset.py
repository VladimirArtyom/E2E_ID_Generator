import pandas as pd
import torch
import random
import spacy
import datasets
from transformers import AutoTokenizer
from typing import List, Mapping, Tuple
from torch.utils.data import Dataset


class QAGDataset(Dataset):
    def __init__(this,
                 data: datasets.Dataset,
                 max_length: int,
                 pad_mask_id: int,
                 tokenizer: AutoTokenizer) -> None:
        this.data = pd.DataFrame(data)
        this.max_length = max_length
        this.pad_mask_id = pad_mask_id
        this.tokenizer = tokenizer

    def _mask_padding_label(this, labels: torch.Tensor) -> torch.Tensor:
        return torch.where(labels == this.tokenizer.pad_token_id, -100, labels)

    def _encode_text(this, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_source = this.tokenizer(text,
                                        max_length=this.max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")

        return (encoded_source['input_ids'].flatten(),
                encoded_source["attention_mask"].flatten())


    def __len__(this):
        return this.data.shape[0]

    def __getitem__(this, index: int) -> Mapping[str, torch.Tensor]:
        item: pd.DataFrame = this.data.iloc[index]

        context = item.context
        answer = item.answer
        context_ids, attention_mask = this._encode_text("<answer> {} <context> {}".format(answer, context))
        labels, _ = this._encode_text(item.question)
        masked_labels = this._mask_padding_label(labels)

        return {
            "input_ids": context_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels.flatten(),
        }

class QAGEvaluatorDataset(Dataset):
    def __init__(this,
                 data: pd.DataFrame,
                 max_length: int,
                 tokenizer: AutoTokenizer,
                 ) -> None:
        this.data = pd.DataFrame(data)[:10]
        this.max_length = max_length
        this.tokenizer = tokenizer
        this.spacy_tokenizer = spacy.load("xx_ent_wiki_sm")
        this.transforms = [this.shuffle, this.corrupt]


    def __len__(this) -> int:
        return len(this.data)

    def _encode_text(this, question: str, answer: str) -> Mapping[str, torch.Tensor]:
        encoded_source = this.tokenizer(text=question,
                                        text_pair=answer,
                                        max_length=this.max_length,
                                        truncation=True,
                                        padding="max_length",
                                        return_tensors="pt"
                                        )
        return encoded_source

    def __getitem__(this, index: int) -> Mapping[str, torch.Tensor]:
        item = this.data.iloc[index]
        question = item.question
        answer = item.answer
        label_choice = random.choice([0, 1])
        if label_choice == 0:
            question, answer = random.choice(this.transforms)(question, answer)

        encoded_input = this._encode_text(question=question,
                                          answer=answer)
        return {
            "input_ids": encoded_input["input_ids"].flatten(),
            "attention_mask": encoded_input["attention_mask"].flatten(),
            "token_type_ids": encoded_input["token_type_ids"].flatten(),
            "labels": torch.tensor(label_choice, dtype=torch.int64),
        }

    def shuffle(this,  question: str, answer) -> Tuple[str, str]:
        shuffled_answer = answer
        while shuffled_answer == answer:
            shuffled_answer = this.data.sample(1)['answer'].item()
        return question, shuffled_answer

    def corrupt(this, question: str, answer: str) -> Tuple[str, str]:
        doc = this.spacy_tokenizer(question)
        if len(doc.ents) > 1:
            copy_ent = str(random.choice(doc.ents))
            for ent in doc.ents:
                question = question.replace(str(ent), copy_ent)
        elif len(doc.ents) == 1:
            answer = str(doc.ents[0])
        else:
            question, answer = this.shuffle(question, answer)
        return question, answer
