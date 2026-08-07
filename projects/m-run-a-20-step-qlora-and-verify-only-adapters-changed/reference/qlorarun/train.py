import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


def run_training_steps(model, tokenizer, dataset, steps=20):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=steps,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    return model
