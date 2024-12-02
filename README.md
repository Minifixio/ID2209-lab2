# ID2209-lab2

# Lab Instructions
## Task 1: Fine-tune a model for language transcription, add a UI
### Instructions
- Fine-tune an existing pre-trained large language model on the FineTome
Instruction Dataset
- Build and run an inference pipeline with a Gradio UI on Hugging Face

### Notes
- You should fine-tune the LLM on the Fine Tome Dataset hosted at Hugging Face.
- You will need to checkpoint the weights periodically, so that you can restart your training from where you left off. Even if you have your own GPU you still have to demonstrate this task.
- You have to save your fine tuned LLM somewhere - e.g., on Hopsworks or Google Drive, so that you can download it for use in your UI.


## Task 2: Improve pipeline scalability and model performance
### Instructions
- Describe in your README.md program ways in which you can improve model performance are using
    1. *model-centric approach* - e.g., tune hyperparameters, change the
    fine-tuning model architecture, etc
    2. *data-centric approach* - identify new data sources that enable you to train a better model that one provided in the blog post If you can show results of improvement, then you get the top grade.

- Try out fine-tuning a couple of different open-source foundation LLMs to get one that works best with your UI for inference (inference will be on CPUs, so big models will be slow).

### Notes
- You are free to use other fine-tuning frameworks, such as Axolotl of HF FineTuning - you do not have to use the provided unsloth notebook.

# Our Work
## Task 1
### Finetuning
We trained the LLM using the finetuning Notebook. In order to save checkpoints and train the models in several runs on one epoch, we changed the following to the trainer `TrainingArguments` :
```python
output_dir = "outputs",
num_train_epochs = 1,
save_steps = 50, # Saves checkpoints every 50 steps
```
so it looks like :
```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50, # Saves checkpoints every 50 steps
        report_to = "none",
    ),
)
```

So the trainer saves checkpoints every 50 steps.

To start the training, use :
```python
trainer_stats = trainer.train()
```

To resume the training from checkpoints, use:
```python
trainer_stats = trainer.train(resume_from_checkpoint=True)
```


## Adding a UI
We decided to add a Gradio UI that allows users to input their personal and job application details to generate a personalized, professional cover letter using a pretrained LLM.

You can find the UI in the `huggingface/app.py` file.

The code integrates a pretrained Llama model using `Llama.from_pretrained()` for chat-based completion, dynamically constructs a prompt in the `generate_cover_letter` function by incorporating user inputs like `name`, `job_title`, `company_name`, `skills`, and `reasons`, and removes repetitive prefixes from the LLM response for cleaner output; the Gradio UI provides text input fields for these details, along with a button to trigger letter generation, dynamically displaying the output in a text box, and uses a customizable system message to guide the LLM's behavior in generating professional cover letters.

You can use it [here](https://huggingface.co/spaces/minifixio/ID2223-lab2).