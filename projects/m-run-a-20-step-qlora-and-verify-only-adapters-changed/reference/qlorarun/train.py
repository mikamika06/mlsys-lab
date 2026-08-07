import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType


def run_qlora_training(model, tokenizer, dataset, steps=20):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=1,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return model
