import torch
import argparse
import os

from freeflux.load_model import load_dev_model_4bit, load_schnell_model_4bit

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using FLUX model')
    parser.add_argument('--prompt', type=str, default="realistic, best quality, extremely detailed, ray tracing, photorealistic." \
                         "A blue cat holding a sign that says hello world",
                        help='The prompt for image generation')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated image')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated image')
    parser.add_argument('--guidance-scale', type=float, default=3.5, help='Guidance scale for generation')
    parser.add_argument('--steps', type=int, default=16, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for generation')
    parser.add_argument('--cpu-offload', action='store_true', help='Enable CPU offload')
    parser.add_argument('--model', type=str, default='dev', choices=['dev', 'schnell'], help='Model to use for generation (dev or 4bit-schnell)')
    parser.add_argument('--lora', type=str, default=None, help='add lora weights into the model')
    parser.add_argument('--lora-scale', type=float, default=1.0, help='add lora weights into the model')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model == 'dev':
        pipe = load_dev_model_4bit()
    else:
        assert args.model == 'schnell'
        pipe = load_schnell_model_4bit()

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to('cuda')

    lora_path = args.lora
    if lora_path:
        weights_name = os.path.basename(lora_path)
        adaptor_name = os.path.splitext(weights_name)[0]
        pipe.load_lora_weights(lora_path, weight_name=weights_name, adaptor_name=adaptor_name)
    with torch.inference_mode():
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            output_type="pil",
            num_inference_steps=args.steps,
            max_sequence_length=512,
            joint_attention_kwargs={"scale": args.lora_scale},
            generator=torch.Generator("cpu").manual_seed(args.seed)
        ).images[0]

    image.show()


if __name__ == "__main__":
    main()
