import transformers
import os
import datetime
import torch

from benchita.task import get_task
from benchita.template import get_template
from benchita.utils import parse_str_args, build_inference_dataset, run_inference
from benchita.logging import log_info, log_warn, log_error
from benchita.dummy import DummyModel

def evaluate(args):
    if args.dummy_run:
        log_warn("Dummy run enabled, the model will not be loaded, instead a dummy model will be used")

    model_cls = getattr(transformers, args.model_class)
    tokenizer_cls = getattr(transformers, args.tokenizer_class)
    task_cls = get_task(args.task)
    dtype = getattr(torch, args.dtype)

    log_info(f"Task: {args.task} (class: {task_cls.__name__})")
    log_info(f"Model: {args.model} (class: {model_cls.__name__})")
    log_info(f"Tokenizer: {args.tokenizer} (class: {tokenizer_cls.__name__})")

    model_args = parse_str_args(args.model_args)
    tokenizer_args = parse_str_args(args.tokenizer_args)
    apply_chat_template_args = parse_str_args(args.apply_chat_template_args)
    generate_args = parse_str_args(args.generate_args)

    log_info(f"Model args: {model_args}")
    log_info(f"Tokenizer args: {tokenizer_args}")
    log_info(f"Apply chat template args: {apply_chat_template_args}")
    log_info(f"Generate args: {generate_args}")

    log_info("Loading tokenizer...")
    tokenizer = tokenizer_cls.from_pretrained(args.tokenizer, padding_side="left", **tokenizer_args)

    if ((hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None) and args.template != "") and not args.force_template:
        log_error("Tokenizer has a chat template, but a template was provided use --force-template to ignore this error")

    if (not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None) and args.template == "":
        log_warn("Tokenizer does not have a chat template and no template was provided, using 'default' template")
        args.template = "default"

    if args.template != "":
        template = get_template(args.template)()
        chat_template = template.apply_chat_template
    else:
        chat_template = tokenizer.chat_template

    if args.patch_tokenizer_pad:
        log_warn("Patching tokenizer to use eos_token as padding token")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        log_error("Tokenizer does not have a pad token, please use a tokenizer with a pad token or use --patch-tokenizer-pad to patch it")

    log_info("Loading task...")
    task = task_cls()

    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < args.max_length + task.max_new_tokens:
        log_error(f"The model max_length ({tokenizer.model_max_length}) is smaller than the sum of the tokenized max_length ({args.max_length}) and the generated max_new_tokens ({task.max_new_tokens}). Consider decreasing the tokenized max_length with --max-length")

    if args.dummy_run:
        log_warn("Loading dummy model...")
        model = DummyModel(task)
    else:
        log_info("Loading model...")
        model = model_cls.from_pretrained(args.model, torch_dtype=dtype, **model_args).to(args.device)

    log_info("Building inference dataset...")
    inference_ds = build_inference_dataset(
        tokenizer=tokenizer,
        task=task,
        num_shots=args.num_shots,
        system_style=args.system_style,
        chat_template=chat_template,
        apply_chat_template_args=apply_chat_template_args,
        max_length=args.max_length
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
            batch_size=args.batch_size,
            generate_args=generate_args,
            device=args.device,
            dry_run=args.dry_run
        )

    log_info("Evaluating results...")
    results = task.evaluate(inference)

    if not args.dry_run and not args.dummy_run:
        if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
        sanitezed_model = args.model.replace("/", "_")
        fname = f"{sanitezed_model}_{args.task}_{datetime.datetime.now().isoformat()}.json"
        fpath = os.path.join(args.save_dir, fname)
        with open(fpath, "w") as f:
            import json
            json.dump(dict(
                args=vars(args),
                results=results,
            ), f, indent=4)
            log_info(f"Results saved to {fpath}")

    summary = task.results_summary(results)
    summary.index = [args.model]

    log_info("Results summary:")
    print(summary)