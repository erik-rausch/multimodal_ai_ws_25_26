import json

import librosa
import random
import soundfile as sf
from pathlib import Path
import torch
import tqdm
from datasets import Audio, Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_utils import ac_tq_instruction, tc_aq_instruction, ac_aq_instruction, ac_tq_audio, tc_aq_audio, ac_aq_audio
from train_utils import evaluate
from transformers import TrainerCallback
from transformers import TrainingArguments, Trainer
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor

TASKS = {
    "ac-aq": {"audio": ac_aq_audio, "instr": ac_aq_instruction},
    "ac-tq": {"audio": ac_tq_audio, "instr": ac_tq_instruction},
    "tc-aq": {"audio": tc_aq_audio, "instr": tc_aq_instruction},
}

train_mode = "mixed"
run_number = 2
model_name = "/training-1/modelhub/granite-speech-3.3-2b"

if train_mode not in TASKS and train_mode != "mixed":
    raise ValueError(f"Unknown train_mode: {train_mode}")

def prep_entry(entry, tokenizer):
    """
    Set per-example task mode and create prompt using the corresponding instruction.
    If train_mode is 'mixed' pick one of the tasks at random for this entry.
    """
    if train_mode == "mixed":
        mode = random.choice(list(TASKS.keys()))
    else:
        mode = train_mode
    entry["task_mode"] = mode
    entry["prompt"] = TASKS[mode]["instr"](tokenizer, entry)
    return entry

output_dir = f"logs/{train_mode}-{run_number}"

if Path(output_dir).exists():
    print(f"Output directory {output_dir} already exists. Skipping training.")
    exit(0)


def load_dataset(path, split):
    entries = []
    with open(f"{path}/{split}.jsonl", encoding="UTF-8") as rf:
        for line in rf:
            entry = json.loads(line.strip())
            entries.append(entry)
    dataset = Dataset.from_list(entries)
    return dataset


def prepare_dataset(ds, processor):
    ds = ds.cast_column(
        "context_audio",
        Audio(sampling_rate=processor.audio_processor.sampling_rate, decode=False)
    )
    ds = ds.cast_column(
        "question_audio",
        Audio(sampling_rate=processor.audio_processor.sampling_rate, decode=False)
    )

    ds = ds.map(prep_entry, fn_kwargs=dict(tokenizer=processor.tokenizer))
    return ds


class GraniteCollator:
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, entries):
        prompts = [example["prompt"] for example in entries]
        audios = []
        for entry in entries:
            mode = entry.get("task_mode", train_mode if train_mode in TASKS else "ac-aq")
            audio_func = TASKS.get(mode, TASKS["ac-aq"])["audio"]
            array, sr = audio_func(entry)  # decode WAV using the chosen task's loader
            if array.ndim > 1:
                array = array.mean(axis=1)  # convert stereo → mono
            audios.append(array.astype("float32"))

        processed = self.processor(prompts, audios, return_tensors="pt", padding=True, padding_side="left")
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None
        # tokenize targets
        if not self.inference_mode:
            targets = [entry["answer_text"] + self.processor.tokenizer.eos_token for entry in entries]
            targets = self.processor.tokenizer(targets, return_tensors="pt", padding=True, padding_side="right")
            # combine prompt+targets
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            labels = targets.input_ids.clone()
            # Set non-target tokens to -100 for loss calculation
            labels[~(targets.attention_mask.bool())] = -100
            labels = torch.cat([torch.full_like(processed.input_ids, -100), labels], dim=1)

        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": processed.input_features,
            "input_features_mask": processed.input_features_mask
        })


def compute_performance(model, processor, cur_dataset, print_limit=5):
    print_limit = min(print_limit, len(cur_dataset))
    with torch.no_grad():

        collator = GraniteCollator(processor, inference_mode=True)
        dataloader = DataLoader(cur_dataset, batch_size=4, collate_fn=collator, num_workers=0)

        all_outputs = []
        for batch in tqdm.tqdm(dataloader, desc="Running inference"):
            batch = batch.to("cuda")
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(**batch, max_new_tokens=256, num_beams=4, early_stopping=True)
            input_length = batch.input_ids.shape[1]
            outputs = outputs[:, input_length:].cpu()
            for x in outputs:
                all_outputs.append(processor.tokenizer.decode(x, skip_special_tokens=True))

        contexts = [x for x in cur_dataset["context_text"]]
        questions = [x for x in cur_dataset["question_text"]]
        answers = [x for x in cur_dataset["answer_text"]]

        lisa_outputs = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}

        # Counter for printed outputs
        printed = 0

        for context, question, answer, output in tqdm.tqdm(
                zip(contexts, questions, answers, all_outputs),
                desc="Generating LISA Judge Outputs",
                total=len(contexts)
        ):
            judge = evaluate(context, question, answer, output)
            lisa_outputs[judge] += 1

            # print only first print_limit examples
            if printed < print_limit:
                print(f"Question: {question}\nOutput {printed + 1}: {output}\nAnswer: {answer}\nJudge: {judge}\n")
                printed += 1

        print(f"""
        Reproduction: {lisa_outputs[0]}
        Correct: {lisa_outputs[1]}
        Wrong: {lisa_outputs[2]}
        Partially: {lisa_outputs[3]}
        Format Error: {lisa_outputs[-1]}
        """)
        return lisa_outputs[1] / len(cur_dataset)


train_dataset = load_dataset("dataset", "train")
val_dataset = load_dataset("dataset", "validate")
test_dataset = load_dataset("dataset", "test").take(30)

processor = GraniteSpeechProcessor.from_pretrained(model_name)
model = GraniteSpeechForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

train_dataset = prepare_dataset(train_dataset, processor)
val_dataset = prepare_dataset(val_dataset, processor)
test_dataset = prepare_dataset(test_dataset, processor)

performance_before_train = compute_performance(model, processor, test_dataset)
print(f"Performance before finetuning {performance_before_train}")


class PerformanceCallback(TrainerCallback):
    def __init__(self, processor, dataset, log_dir=output_dir, val_size=2):
        super().__init__()
        self.processor = processor
        self.full_dataset = dataset
        self.writer = SummaryWriter(log_dir)
        self.val_size = val_size

    def on_epoch_end(self, args, state, control, **kwargs):
        indices = random.sample(range(len(self.full_dataset)), self.val_size)
        subset = self.full_dataset.select(indices)
        model = kwargs["model"]
        model.eval()
        with torch.no_grad():
            perf = compute_performance(model, self.processor, subset)
        print(f"Performance after epoch {state.epoch}: {perf}")

        self.writer.add_scalar("Accuracy", perf, int(state.epoch))
        self.writer.flush()


for n, p in model.named_parameters():
    # tranining only the projector/lora layers
    p.requires_grad = "projector" in n or "lora" in n

args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    report_to="tensorboard",
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataloader_num_workers=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=48,
    num_train_epochs=9.0,
    warmup_ratio=0.15,
    logging_steps=0.1,
    learning_rate=3e-4,
    data_seed=42,
    save_total_limit=1,
)

data_collator = GraniteCollator(processor)
performance_callback = PerformanceCallback(processor, val_dataset, log_dir=output_dir, val_size=100)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    callbacks=[performance_callback]
)
trainer.train()

performance_after_train = compute_performance(model, processor, test_dataset)
print(f"Performance after finetuning {performance_after_train}")
