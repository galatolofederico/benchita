import transformers
import os
import datetime
import torch

from benchita.task import get_task
from benchita.template import get_template
from benchita.utils import build_inference_dataset, run_inference
from benchita.logging import log_info, log_warn, log_error
from benchita.dummy import DummyModel

def evaluate(*, job, args, results_file, device="cpu"):
    if args.dummy_run:
        log_warn("Dummy run enabled, the model will not be loaded, instead a dummy model will be used")

    model_config = job["model"]
    task_config = job["task"]

    model_cls = getattr(transformers, model_config.model.class_name)
    tokenizer_cls = getattr(transformers, model_config.tokenizer.class_name)
    task_cls = get_task(task_config.name)

    log_info(f"Task: {task_config.name} (class: {task_cls.__name__})")
    log_info(f"Model: {model_config.model.class_name} (class: {model_cls.__name__})")
    log_info(f"Tokenizer: {model_config.tokenizer.class_name} (class: {tokenizer_cls.__name__})")

    log_info(f"Model args: {model_config.model.args}")
    log_info(f"Tokenizer args: {model_config.tokenizer.args}")
    log_info(f"Template args: {model_config.template.args}")
    log_info(f"Generate args: {model_config.generate.args}")

    log_info("Loading tokenizer...")
    tokenizer = tokenizer_cls.from_pretrained(model_config.tokenizer.name, padding_side="left", **model_config.tokenizer.args)

    if ((hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None) and model_config.template.name is not None) and not model_config.template.force:
        log_error("Tokenizer has a chat template, but a template was provided, set force=True in the temaplte config to ignore this error")

    if (not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None) and model_config.template.name is None:
        log_warn("Tokenizer does not have a chat template and no template was provided, using 'default' template")
        model_config.template.name = "default"

    if model_config.template.name is not None:
        template = get_template(model_config.template.name)()
        chat_template = template.apply_chat_template
    else:
        chat_template = tokenizer.apply_chat_template
    
    if model_config.tokenizer.patch_tokenizer_pad:
        log_warn("Patching tokenizer to use eos_token as padding token")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        log_error("Tokenizer does not have a pad token, please use a tokenizer with a pad token or specify patch_tokenizer_pad=True in the tokenizer config to patch it")

    log_info("Loading task...")
    task = task_cls()

    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < model_config.tokenizer.max_length + task.max_new_tokens:
        log_error(f"The model max_length ({tokenizer.model_max_length}) is smaller than the sum of the tokenized max_length ({model_config.tokenizer.max_length}) and the generated max_new_tokens ({task.max_new_tokens}). Consider decreasing the tokenized max_length setting max_length in the tokenizer config")
    
    if args.dummy_run:
        log_warn("Loading dummy model...")
        model = DummyModel(task)
    else:
        log_info("Loading model...")
        model = model_cls.from_pretrained(
            model_config.model.name,
            torch_dtype=getattr(torch, model_config.model.dtype),
            **model_config.model.args
        ).to(device)

    log_info("Building inference dataset...")
    inference_ds = build_inference_dataset(
        tokenizer=tokenizer,
        task=task,
        num_shots=task_config.num_shots,
        system_style=model_config.template.system_style,
        chat_template=chat_template,
        apply_chat_template_args=model_config.template.args,
        max_length=model_config.tokenizer.max_length
    )

    if args.dry_run:
        log_warn("Dry run enabled, running inference on just one batch")

    if args.dummy_run:
        inference = model.simulate_inference(inference_ds)
    else:
        log_info("Running inference...")
        inference = run_inference(
            dataset=inference_ds,
            model=model,
            tokenizer=tokenizer,
            task=task,
            batch_size=model_config.generate.batch_size,
            generate_args=model_config.generate.args,
            device=device,
            dry_run=args.dry_run
        )

    log_info("Evaluating results...")
    results = task.evaluate(inference)

    if not args.dry_run and not args.dummy_run:
        with open(results_file, "w") as f:
            import json
            json.dump(dict(
                model=model_config.model_dump(),
                task=task_config.model_dump(),
                results=results,
            ), f, indent=4)
            log_info(f"Results saved to {results_file}")

    summary = task.results_summary(results)
    summary.index = [model_config.model.name]

    log_info("Results summary:")
    print(summary)