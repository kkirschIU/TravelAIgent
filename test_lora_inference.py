from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

MODEL_DIR = "outputs/travelaigent-qlora"

def main():
    device = "cuda"
    print("Nutze Device:", device)

    # Tokenizer bevorzugt aus deinem Modellordner laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto"  # hier darf offloading wieder rein
    )

    model.eval()

    prompt = (
        "Du bist TravelAIgent, ein KI-Reiseberater für Island.\n\n"
        "Plane eine 4-tägige Reise im Winter für zwei Personen mit Fokus auf Südküste, "
        "Nordlichter (wenn möglich) und einem Budget von 1500 Euro."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
