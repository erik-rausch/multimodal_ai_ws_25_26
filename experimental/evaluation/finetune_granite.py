from datasets import Audio, Dataset
import json
import torch
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
from transformers.feature_extraction_utils import BatchFeature
from torch.utils.data import DataLoader
import tqdm
import requests
from transformers import TrainingArguments, Trainer
import soundfile as sf
from statistics import mean
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

lisa_prompt = """
Du bist ein Evaluator für ein Spoken-Question-Answering-System.
Du erhältst:
– den transkribierten Kontext des Audios
– die Frage
– die Antwort des Modells
– die Ground-Truth-Antwort

Bewerte ausschließlich anhand des Inhalts, ob die Modellantwort die gleiche Bedeutung wie die Ground-Truth-Antwort hat. Bewerte die Modellantwort niedriger wenn etwas anderes als ausschließlich eine korrekte Antwort zurückgegeben wird! Bewerte Antworten mit Artefakten niedriger, z.B. wenn das Modell 'Beantworte die Frage' wiederholt.
Wenn das Modell 1:1 den Text transkribiert, bewerte deutlich niedriger!
Synonyme und Paraphrasen zählen als korrekt.
Wenn die Modellantwort Informationen erfindet oder falsche Details enthält, bewerte niedriger.
Nutze den Kontext nur, um Halluzinationen zu erkennen, aber orientiere dich inhaltlich primär an der Ground-Truth.

Gib ausschließlich eine einzige Zahl von 1 bis 10 als Output aus — ohne Text, ohne Erklärung.

Bewertungsskala (nur Zahl ausgeben):
1 = völlig falsch
5 = teilweise korrekt
10 = bedeutungsidentisch / perfekt korrekt

Eingaben:
KONTEXT: {text}
FRAGE: {question}
MODELLANTWORT: {output}
GROUND-TRUTH: {answer}

Gib jetzt nur die Zahl aus.
"""

api_token = None
url = None

with open("api_token.txt", encoding="UTF-8") as rf:
    api_token = rf.read().strip()

with open("url.txt", encoding="UTF-8") as rf:
    url = rf.read().strip()


def load_dataset(path, split):
    entries = []
    with open(f"{path}/{split}.jsonl", encoding="UTF-8") as rf:
        for line in rf:
            entry = json.loads(line.strip())
            entries.append(entry)
    dataset = Dataset.from_list(entries)
    return dataset

def prep_example(example, tokenizer):
    instruction = f"Beantworte die Frage '{example["question"]}' aus dem Inhalt des folgenden Audios:<|audio|>"
    chat = [dict(role="user", content=instruction)]
    example["prompt"] = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
    )
    return example

def prepare_dataset(ds, processor):
    ds = ds.cast_column(
    "audio",
    Audio(
        sampling_rate=processor.audio_processor.sampling_rate,
        decode=False            # <-- IMPORTANT
    )
    )
    ds = ds.map(prep_example,
        fn_kwargs=dict(tokenizer=processor.tokenizer),
    )
    return ds


class GraniteCollator:
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, examples):
        prompts = [example["prompt"] for example in examples]
        audios = []
        for ex in examples:
            audio_info = ex["audio"]
            path = audio_info["path"]
            array, sr = sf.read(path)      # decode WAV
            if array.ndim > 1:
                array = array.mean(axis=1)  # convert stereo → mono
            audios.append(array.astype("float32"))
        processed = self.processor(prompts, audios, return_tensors="pt", padding=True, padding_side="left")
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None
        # tokenize targets
        if not self.inference_mode:
            targets = [example["answer"] + self.processor.tokenizer.eos_token for example in examples]
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

def compute_performance(model, processor, cur_dataset):
    with torch.no_grad():
        def ask_lisa(question: str) -> str:
            global url
            global api_token

            model: str = "lisa-v40-rc2-gpt-oss120b"
            headers = {'Authorization': f'Bearer {api_token}',
                        "Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": [
                {
                  "role": "user",
                  "content": question
                }
              ]
            }
            response = requests.post(url, json=data, headers=headers)

            data = response.json()["choices"][0]["message"]["content"]

            return str(data)
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
        texts = [x for x in cur_dataset["text"]]
        questions = [x for x in cur_dataset["question"]]
        answers = [x for x in cur_dataset["answer"]]
        lisa_outputs = []
        for text, question, answer, output in tqdm.tqdm(zip(texts, questions, answers, all_outputs), desc="Generating LISA Judge Outputs", total=len(texts)):
            lisa_question = lisa_prompt.format(
                text=text,
                question=question,
                answer=answer,
                output=output
            )
            lisa_output = ask_lisa(lisa_question)
            try:
                lisa_output_number = float(lisa_output)
                lisa_outputs.append(lisa_output_number)
            except Exception as ex:
                print(f"Could not cast lisa output to integer: {lisa_output}! Adding 0 to evaluation.")
                lisa_outputs.append(0.0)

        performance = mean(lisa_outputs)

        return performance


train_dataset = load_dataset("dataset", "train_nodev")
val_dataset = load_dataset("dataset", "train_dev")
test_dataset = load_dataset("dataset", "test").take(30)

model_name = "/training-1/modelhub/granite-speech-3.3-2b"
processor = GraniteSpeechProcessor.from_pretrained(model_name)
model = GraniteSpeechForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

train_dataset = prepare_dataset(train_dataset, processor)
val_dataset = prepare_dataset(val_dataset, processor)
test_dataset = prepare_dataset(test_dataset, processor)

performance_before_train = compute_performance(model, processor, test_dataset)
print(f"Performance before finetuning {performance_before_train}")

class PerformanceCallback(TrainerCallback):
    def __init__(self, processor, dataset, log_dir="runs/performance"):
        super().__init__()
        self.processor = processor
        self.dataset = dataset
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        with torch.no_grad():
            perf = compute_performance(model, self.processor, self.dataset)
        print(f"Performance after epoch {state.epoch:.0f}: {perf}")
        self.writer.add_scalar("Performance/test_dataset", perf, int(state.epoch))
        self.writer.flush()

for n, p in model.named_parameters():
    # tranining only the projector/lora layers
    p.requires_grad = "projector" in n or "lora" in n
output_dir = "logs"
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
    gradient_accumulation_steps=8,
    num_train_epochs=15.0,
    warmup_ratio=0.1,
    logging_steps=0.1,
    learning_rate=5e-5,
    data_seed=42,
    save_total_limit=1
)

data_collator = GraniteCollator(processor)
performance_callback = PerformanceCallback(processor, test_dataset, log_dir=output_dir)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    #callbacks=[performance_callback]
)
trainer.train()


performance_after_train = compute_performance(model, processor, test_dataset)
print(f"Performance after finetuning {performance_after_train}")
