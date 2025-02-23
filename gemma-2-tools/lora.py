import os
import json
import argparse
import torch
from transformers.trainer import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from trl import SFTTrainer
from datasets import Dataset, load_dataset
from accelerate import Accelerator

def main(args):
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ----------------------------------------------------
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_modelname,
        attn_implementation="eager",
        add_eos_token=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # ----------------------------------------------------
    # Define data format
    PROMPT_FORMAT = """<start_of_turn>user
    {instruction}
    <end_of_turn>
    <start_of_turn>model
    {output}
    <end_of_turn>
    """

    def format_data(line):
        prompt = line["prompt"]
        output = line["response"]
        full_prompt = PROMPT_FORMAT.format(instruction=prompt, output=output)
        return {"text": full_prompt}

    # Data setup
    train_filename = args.train_data
    train_dataset = load_dataset("json", data_files=train_filename, cache_dir=args.cache_dir, split="train")
    train_dataset = train_dataset.map(format_data)


    # ----------------------------------------------------
    # Model setup
    device_map = "auto"

    batch_size = args.total_batch_size
    gradient_accumulation_steps = batch_size // (args.per_device_batch_size)
    
    #ddpをonにするとOOMが起きる．onにした場合の挙動は未検証なのでdatalodaerやgasの設定に注意
    ddp = False #world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    elif world_size > 1:
        #accelerateからの起動
        device_map ={"": Accelerator().process_index}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Set model_dtype
    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = torch.float16

    fp16_flag = False
    bf16_flag = False
    if args.model_dtype == "fp16":
        fp16_flag = True
    elif args.model_dtype == "bf16":
        bf16_flag = True


    # ----------------------------------------------------
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_modelname,
        device_map="auto",
        torch_dtype=model_dtype
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules
    )
    model = get_peft_model(model, peft_config)
    
    torch.cuda.empty_cache()
    model.config.use_cache = False


    # ----------------------------------------------------
    # Resume
    resume_from_checkpoint = args.resume_dirname
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        ) # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            ) # Only LoRA model - LoRA config above has to fit
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")


    # ----------------------------------------------------
    # Set params

    # fp16指定時に処理が止まらないようパラメータを設定
    # 詳細：https://huggingface.co/docs/peft/v0.11.0/en/developer_guides/troubleshooting#valueerror-attempting-to-unscale-fp16-gradients
    if fp16_flag:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    model.print_trainable_parameters()

    # training
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1  gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        #args.per_device_batch_size *=torch.cuda.device_count()
    if ddp:
        model =model.to(local_rank)
        model =DDP(model, device_ids =[local_rank])

    # Set params
    learning_rate = args.lr
    warmup_steps = args.warmup_steps
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    epochs = args.epochs

    save_model_dir = args.save_dirname
    max_steps = int(args.num_train_data/batch_size * epochs)
    local_rank = args.local_rank

    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16_flag,
        bf16=bf16_flag,
        logging_steps=logging_steps,
        logging_dir="./tensorboard_logs",
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=save_model_dir,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=True,
        report_to="tensorboard",
        local_rank=local_rank,
        lr_scheduler_type=args.lr_scheduler_type,
        dataloader_num_workers=args.dataloader_num_workers
    )

    if world_size == 1 or Accelerator().process_index == 0:
        print(training_arguments)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        args=training_arguments,
        max_seq_length=args.max_seq_len
    )


    # ----------------------------------------------------
    # Train model
    print("Ready to peft.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)

    print("Finished training.")


def get_dtype(dtype: str):
    if dtype == "fp32":
        return torch.float32
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f"dtype {dtype} is not supported. "+ f"We only support fp32, fp16, and bf16 currently."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning Llama-3.1-Swallow-8B-Instruct-v0.3.", add_help=True)

    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--num_train_data", required=True, type=int)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--per_device_batch_size", required=True, type=int)
    parser.add_argument("--total_batch_size", required=True, type=int)
    parser.add_argument("--lora_r", required=False, type=int)
    parser.add_argument("--lora_alpha", required=False, type=int)
    parser.add_argument("--base_modelname", required=False, type=str)
    parser.add_argument("--save_dirname", required=True, type=str)
    parser.add_argument("--resume_dirname", required=False, type=str, default=None)

    parser.add_argument("--local_rank", required=False, type=int, default=0)
    parser.add_argument("--target_modules", required=False, type=str, nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--lr", required=False, type=float, default=3e-4)
    parser.add_argument("--warmup_steps", required=False, type=int, default=0)
    parser.add_argument("--logging_steps", required=False, type=int, default=20)
    parser.add_argument("--save_steps", required=False, type=int, default=100)
    parser.add_argument("--lr_scheduler_type", required=False, type=str, default="cosine")
    parser.add_argument("--max_seq_len", required=False, type=int, default=2048)
    parser.add_argument("--epochs", required=False, type=int, default=1)
    parser.add_argument("--dataloader_num_workers", required=False, type=int, default=0)
    parser.add_argument("--model_dtype", required=False, type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save_total_limit", required=False, type=int, default=10)

    args = parser.parse_args()
    main(args)