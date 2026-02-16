import argparse
import os
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np


def preprocess_function(examples, tokenizer, max_length=256):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--val', default='data/val.csv')
    parser.add_argument('--model_name', default='uer/bert-base-chinese-cluecorpussmall')
    parser.add_argument('--output_dir', default='models/violation_classifier')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    # 加载数据
    data_files = {'train': args.train, 'validation': args.val}
    ds = load_dataset('csv', data_files=data_files)

    # tokenizer 与模型
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    tokenized_train = ds['train'].map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val = ds['validation'].map(lambda x: preprocess_function(x, tokenizer), batched=True)

    tokenized_train = tokenized_train.rename_column('label', 'labels')
    tokenized_val = tokenized_val.rename_column('label', 'labels')
    tokenized_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print('训练完成，模型已保存到', args.output_dir)

if __name__ == '__main__':
    main()
