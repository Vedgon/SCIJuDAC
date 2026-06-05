"""
Configures PeFT and 4-bit quantization for LLMs.
"""
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

def quantization_config() -> tuple[BitsAndBytesConfig, LoraConfig]:
	"""
	Configures LoRA and BitsandBytes for PeFT and 4-bit
	quantization.

	Arguments
	---------
	None

	Returns
	-------
	bnb_config : BitsAndBytesConfig
		4-bit quantization configuration object.

	peft_config : LoraConfig
		LoRA configuration object for PeFT.
	"""
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
	)

	peft_config = LoraConfig(
		r=16,
		lora_alpha=32,
		target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 
		lora_dropout=0.05,
		bias="none",
		task_type="SEQ_CLS",
	)

	return bnb_config, peft_config