import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE = "data/train.jsonl"
OUTPUT_DIR = "outputs/travelaigent-qlora"


def main():
    # 1. Dataset laden
    print(f"Lade Trainingsdaten aus {TRAIN_FILE} ...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

    # 2. Tokenizer laden
    print("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Sicherstellen, dass ein pad_token existiert
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 4-Bit-Quantisierung konfigurieren
    print("Konfiguriere 4-Bit-Quantisierung...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 4. Basismodell laden
    print("Lade Basismodell...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Für QLoRA vorbereiten
    print("Bereite Modell für k-Bit-Training vor...")
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA-Konfiguration anwenden
    print("Erzeuge und wende LoRA-Konfiguration an...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # Optional könnten hier target_modules explizit gesetzt werden
    )
    model = get_peft_model(model, peft_config)

    # 6. Dataset → Token-Format bringen
    # Wir wandeln die Chat-Struktur in reinen Text um und tokenisieren diesen.
    def preprocess(example):
        messages = example["messages"]
        # Chat-Template nutzen, um aus messages einen String zu bauen
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=2048,
        )
        return tokenized

    print("Formatiere und tokenisiere Trainingsbeispiele...")
    tokenized_dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
    )

    # 7. Data Collator für Language Modeling (kein Masked LM, sondern Causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 8. TrainingArguments – vorsichtig dimensioniert für 8 GB VRAM
    print("Setze TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1.0,
        logging_steps=10,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit",
        report_to=[],
    )

    # 9. Trainer initialisieren
    print("Initialisiere Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 10. Training starten
    print("Starte Training...")
    trainer.train()

    # 11. LoRA-Adapter + Tokenizer speichern
    print(f"Speichere LoRA-Adapter und Tokenizer nach {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fertig.")


if __name__ == "__main__":
    main()
