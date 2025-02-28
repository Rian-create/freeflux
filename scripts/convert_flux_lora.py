import argparse
from safetensors.torch import save_file
from safetensors import safe_open
from freeflux.lora import convert_ai_toolkit_to_sd_scripts, convert_sd_scripts_to_ai_toolkit
import logging

logger = logging.getLogger(__name__)

def main(args):
    # load source safetensors
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    logger.info(f"Converting {args.src} to {args.dst} format")
    if args.src == "ai-toolkit" and args.dst == "sd-scripts":
        state_dict = convert_ai_toolkit_to_sd_scripts(state_dict)
    elif args.src == "sd-scripts" and args.dst == "ai-toolkit":
        state_dict = convert_sd_scripts_to_ai_toolkit(state_dict)

        # eliminate 'shared tensors' 
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone()
    else:
        raise NotImplementedError(f"Conversion from {args.src} to {args.dst} is not supported")

    # save destination safetensors
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA format")
    parser.add_argument("--src", type=str, default="sd-scripts", help="source format, ai-toolkit or sd-scripts")
    parser.add_argument("--dst", type=str, default="ai-toolkit", help="destination format, ai-toolkit or sd-scripts")
    parser.add_argument("--src_path", type=str, default=None, help="source path")
    parser.add_argument("--dst_path", type=str, default=None, help="destination path")
    args = parser.parse_args()
    main(args)
