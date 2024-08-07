import os
import json
import random
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from evaluate import load
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)


class VITFineTuner:
    def __init__(
        self,
        dataset_path,
        model_name_or_path="google/vit-base-patch16-224-in21k",
        output_dir="./vit-model-fine-tuned",
    ):
        self.dataset_path = dataset_path
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.metric = load("accuracy")
        self.model = None
        self.training_args = None
        self.trainer = None

    def load_dataset(self):
        train_folder = os.path.join(self.dataset_path, "train")
        test_folder = os.path.join(self.dataset_path, "test")

        # Check if "test" folder exists and contains files
        if os.path.exists(test_folder) and any(os.listdir(test_folder)):
            print(
                "'test' folder already exists and contains files. Skipping data restructuring."
            )
        else:
            # Create "test" folder if it doesn't exist
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
                print(f"Created 'test' folder at {test_folder}")

            # Read the "train" folder in the dataset_path
            for category in os.listdir(train_folder):
                category_path = os.path.join(train_folder, category)

                # Skip if it's not a directory
                if not os.path.isdir(category_path):
                    continue

                # Create a folder with the same name in the "test" folder
                test_category_path = os.path.join(test_folder, category)
                if not os.path.exists(test_category_path):
                    os.makedirs(test_category_path)
                    print(f"Created category folder '{category}' in 'test' folder")

                # Get list of all files in the current category
                files = os.listdir(category_path)
                files_to_move = random.sample(
                    files, int(0.2 * len(files))
                )  # 20% of files

                # Move 20% of the files to the test folder
                for file_name in files_to_move:
                    src_path = os.path.join(category_path, file_name)
                    dst_path = os.path.join(test_category_path, file_name)
                    shutil.move(src_path, dst_path)
                    print(f"Moved {src_path} to {dst_path}")

            print("Dataset restructured successfully")
        self.ds = load_dataset(path=self.dataset_path)
        print("Dataset loaded successfully")

    def transform(self, example_batch):
        inputs = self.processor(
            [x for x in example_batch["image"]], return_tensors="pt"
        )
        inputs["labels"] = example_batch["label"]
        return inputs

    def prepare_dataset(self):
        self.prepared_ds = self.ds.with_transform(self.transform)
        print("Dataset prepared successfully")

    def collate_fn(self, batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    def compute_metrics(self, p):
        return self.metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def setup_model(self):
        labels = self.ds["train"].features["label"].names
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        )
        print("Model setup successfully")

    def setup_training_args(self):
        self.training_args = TrainingArguments(
            output_dir="./tmp-training",
            per_device_train_batch_size=20,
            evaluation_strategy="steps",
            num_train_epochs=10,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            metric_for_best_model="accuracy",
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
        )
        print("Training arguments set up successfully")

    def setup_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.prepared_ds["train"],
            eval_dataset=self.prepared_ds["test"],
            tokenizer=self.processor,
        )
        print("Trainer setup successfully")

    def train(self):
        train_results = self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()
        print("Training completed successfully")

    def plot_metrics(self):
        trainer_state_path = os.path.join(
            self.training_args.output_dir, "trainer_state.json"
        )
        with open(trainer_state_path, "r") as file:
            trainer_state = json.load(file)

        epochs = [
            entry["epoch"] for entry in trainer_state["log_history"] if "loss" in entry
        ]
        losses = [
            entry["loss"] for entry in trainer_state["log_history"] if "loss" in entry
        ]
        learning_rates = [
            entry["learning_rate"]
            for entry in trainer_state["log_history"]
            if "learning_rate" in entry
        ]

        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, losses, label="Loss", marker="o")
        plt.xlabel("Epoch or Step")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs or Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig("./training_loss.jpg")
        plt.close()

        # Plot Learning Rate
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, learning_rates, label="Learning Rate", marker="o")
        plt.xlabel("Epoch or Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule over Epochs or Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig("./learning_rate.jpg")
        plt.close()

        print("Metrics plotted and saved successfully")
