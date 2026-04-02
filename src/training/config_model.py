from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

def model_config():
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
	)

	peft_config = LoraConfig(
		r=8,
		lora_alpha=32,
		target_modules=["q_proj", "v_proj"], # , "k_proj", "o_proj"
		lora_dropout=0.05,
		bias="none",
		task_type="SEQ_CLS",
	)

	return bnb_config, peft_config