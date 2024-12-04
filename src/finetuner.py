from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from dataset_handler import get_training_and_validation_datasets

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True,
        padding="max_length",
        max_length=512)

def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def fine_tune_model(json_path: str, output_dir: str):
    train_dataset, eval_dataset = get_training_and_validation_datasets(json_path)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

    # Prepare PyTorch datasets
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        push_to_hub=False,
        use_cpu=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer
    )

    # Start training
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)