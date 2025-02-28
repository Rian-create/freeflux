import torch
import os
from .model import T5EncoderModel, FluxTransformer2DModel
from diffusers import FluxPipeline

def load_dev_model_4bit():
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    return pipe


def load_schnell_model_4bit():
    # TODO: this does not work at all, due to complex dependencies FIX THIS
    # The issue is mainly due to the quanto lib does not work well with torch!!
    # The model loading "pipe = pipe.to('cuda')" is very slow!!
    # almost stuck in the code below
    #   unpacked_data = torch._weight_int4pack_mm(identity, self._data, group_size, id_scale_and_shift)

    from optimum.quanto import freeze, qfloat8, quantize
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
    dtype = torch.bfloat16
    bfl_repo = "black-forest-labs/FLUX.1-schnell"
    revision = "refs/pr/1"
    local_path = "FLUX.1-schnell-4bit"
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)

    # for torch 2.6 security measures
    text_encoder_2 = torch.load(local_path + '/text_encoder_2.pt', weights_only=False) # not secure only do it when you trust the file
    tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = torch.load(local_path + '/transformer.pt', weights_only=False) # not secure only do it when you trust the file

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    return pipe

def load_lora(pipeline, lora_path, alpha=0.75):
    if lora_path:
        weights_name = os.path.basename(lora_path)
        adapter_name = os.path.splitext(weights_name)[0]
        pipeline.load_lora_weights(lora_path, weight_name=weights_name, adapter_name=adapter_name)
    return pipeline

def unload_lora(pipeline):
    pipeline.unload_lora_weights()
    return pipeline

