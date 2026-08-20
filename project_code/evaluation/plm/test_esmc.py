import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def main():
    cdrs = "GYSITSDYAISYSGSTARGGTGFDYENVDTYGASGQSYSYPLT"

    # optionally use "biohub/ESMC-600M" or "biohub/ESMC-300M"
    model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-300M", device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")

    inputs = tokenizer(cdrs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    penultimate_layer_embed = outputs.hidden_states[-2]

    #print(f"logits shape: {tuple(output.logits.shape)}")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(tokens)
    print(len(tokens))
    print(f"penultimate layer embed shape: {tuple(penultimate_layer_embed.shape)}")

    residue_embeddings = penultimate_layer_embed[:, 1:-1, :]
    print(f"residue embeddings shape: {tuple(residue_embeddings.shape)}")


if __name__ == "__main__":
    main()