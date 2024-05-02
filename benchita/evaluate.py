import transformers
import os
import datetime
import torch

from benchita.task import get_task
from benchita.template import get_template
from benchita.utils import build_inference_dataset, run_inference
from benchita.logging import log_info, log_warn, log_error
from benchita.dummy import DummyModel

def evaluate(*, job, args, results_file, device, worker_id=0):
    if args.dummy_run:
        log_warn("Dummy run enabled, the model will not be loaded, instead a dummy model will be used")

    model_config = job["model"]
    task_config = job["task"]

    model_cls = getattr(transformers, model_config.model.class_name)
    tokenizer_cls = getattr(transformers, model_config.tokenizer.class_name)
    task_cls = get_task(task_config.name)

    log_info(f"Task: {task_config.name} (class: {task_cls.__name__})", worker_id=worker_id)
    log_info(f"Model: {model_config.model.class_name} (class: {model_cls.__name__})", worker_id=worker_id)
    log_info(f"Tokenizer: {model_config.tokenizer.class_name} (class: {tokenizer_cls.__name__})", worker_id=worker_id)

    log_info(f"Model args: {model_config.model.args}", worker_id=worker_id)
    log_info(f"Tokenizer args: {model_config.tokenizer.args}", worker_id=worker_id)
    log_info(f"Template args: {model_config.template.args}", worker_id=worker_id)
    log_info(f"Generate args: {model_config.generate.args}", worker_id=worker_id)

    log_info("Loading tokenizer...", worker_id=worker_id)
    tokenizer = tokenizer_cls.from_pretrained(model_config.tokenizer.name, padding_side="left", **model_config.tokenizer.args)

    if ((hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None) and model_config.template.name is not None) and not model_config.template.force:
        log_error("Tokenizer has a chat template, but a template was provided, set force=True in the temaplte config to ignore this error", worker_id=worker_id)

    if (not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None) and model_config.template.name is None:
        log_warn("Tokenizer does not have a chat template and no template was provided, using 'default' template", worker_id=worker_id)
        model_config.template.name = "default"

    if model_config.template.name is not None:
        template = get_template(model_config.template.name)()
        chat_template = template.apply_chat_template
    else:
        chat_template = tokenizer.apply_chat_template
    
    if model_config.tokenizer.patch_tokenizer_pad:
        log_warn("Patching tokenizer to use eos_token as padding token", worker_id=worker_id)
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        log_error("Tokenizer does not have a pad token, please use a tokenizer with a pad token or specify patch_tokenizer_pad=True in the tokenizer config to patch it", worker_id=worker_id)

    log_info("Loading task...", worker_id=worker_id)
    task = task_cls()
    
    if args.dummy_run:
        log_warn("Loading dummy model...", worker_id=worker_id)
        model = DummyModel(task)
    else:
        log_info("Loading model...", worker_id=worker_id)
        if "load_in_8bit" in model_config.model.args and model_config.model.args["load_in_8bit"]:
            log_warn("bitsandbytes does not support parallel inference, setting device_map to auto")
            if not args.no_parallel:
                log_error("bitsandbytes does not support parallel inference, run benchita with --no-parallel")
            model = model_cls.from_pretrained(
                model_config.model.name,
                device_map="auto",
                **model_config.model.args
            )
        else:
            model = model_cls.from_pretrained(
                model_config.model.name,
                torch_dtype=getattr(torch, model_config.model.dtype),
                **model_config.model.args
            ).to(device)
        if model_config.peft is not None:
            try:
                import peft
            except:
                log_error("Peft model specified in config but not peft module installed, please `pip install peft`", worker_id=worker_id)
            peft_cls = getattr(peft, model_config.peft.class_name)
            log_info(f"Peft: {model_config.peft.class_name} (class: {peft_cls.__name__})", worker_id=worker_id)
            log_info(f"Peft args: {model_config.peft.args}", worker_id=worker_id)
            model = peft_cls.from_pretrained(model, model_config.peft.name, **model_config.peft.args)

    log_info("Building inference dataset...", worker_id=worker_id)
    inference_ds = build_inference_dataset(
        tokenizer=tokenizer,
        task=task,
        num_shots=task_config.num_shots,
        system_style=model_config.template.system_style,
        chat_template=chat_template,
        apply_chat_template_args=model_config.template.args,
        worker_id=worker_id
    )

    if args.dry_run:
        log_warn("Dry run enabled, running inference on just one batch", worker_id=worker_id)

    if args.dummy_run:
        inference = model.simulate_inference(inference_ds)
    else:
        log_info("Running inference...", worker_id=worker_id)
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

    log_info("Evaluating results...", worker_id=worker_id)
    results = task.evaluate(inference)

    summary = task.results_summary(results)
    model_name = model_config.model.name + ("_" + model_config.peft.name) if model_config.peft is not None else ""
    summary.index = [model_name]
    
    log_info("Results summary:", worker_id=worker_id)
    print(summary)
    
    inference_outputs = [
        dict(
            prompt=elem["input"],
            model_input=elem["model_input"],
            output=elem["output"],
        ) for elem in inference
    ]
    final_results = dict(
        model=model_config.model_dump(),
        task=task_config.model_dump(),
        results=results,
        summary=summary.to_json(),
        outputs=inference_outputs,
    )

    if not args.dry_run and not args.dummy_run:
        with open(results_file, "w") as f:
            import json
            json.dump(final_results, f, indent=4)
            log_info(f"Results saved to {results_file}", worker_id=worker_id)

    return final_results