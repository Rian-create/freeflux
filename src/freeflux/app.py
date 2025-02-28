import torch
import gradio as gr
from functools import partial
from freeflux.load_model import load_dev_model_4bit, load_lora, unload_lora
import os
import argparse
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from freeflux.prompts_gen import PromptGenerator
from functools import lru_cache
from PIL import Image
from pathlib import Path
from safetensors import safe_open
from .lora import convert_sd_scripts_to_ai_toolkit

# Update this function to support recursive search
def get_available_loras():
    """Recursively get available LoRA files from the lora directory and its subdirectories."""
    lora_dir = os.path.join(os.getcwd(), "lora")
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir)
    
    lora_files = []
    # Walk through all directories recursively
    for root, _, files in os.walk(lora_dir):
        for file in files:
            if file.endswith('.safetensors'):
                # Get relative path from the current working directory
                rel_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                lora_files.append(rel_path)
    
    # Add empty option as first choice
    choices = [""] + lora_files
    return choices

def refresh_lora_choices():
    """Get available LoRA files and return as update object."""
    choices = get_available_loras()
    return gr.update(choices=choices)

def load_output_images(output_dir):
    """Load all images from the output directory."""
    if not os.path.exists(output_dir):
        return [], []
    
    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(reverse=True)  # Most recent first
    
    images = []
    image_paths = []
    for img_file in image_files:
        img_path = os.path.join(output_dir, img_file)
        try:
            img = Image.open(img_path)
            images.append(img)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return images, image_paths

def get_image_metadata(image_path, output_dir):
    """Get metadata for an image if available."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    metadata_path = os.path.join(output_dir, f"{base_name.split('_seed')[0]}_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def delete_image(image_path, output_dir):
    """Delete an image and its associated metadata file."""
    if not image_path or not os.path.exists(image_path):
        return False, "No image selected or file not found"
    try:
        # Delete the image file
        os.remove(image_path)
        return True, f"Deleted {os.path.basename(image_path)}"
    except Exception as e:
        return False, f"Error deleting image: {str(e)}"

@lru_cache(maxsize=5)
def load_lora_file(lora_path) -> Dict[str, torch.Tensor]:
    state_dict = {}
    is_unet_format = False
    with safe_open(lora_path, framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if "unet" in k:
                is_unet_format = True
            if not ("unet" in k or "transformer" in k):
                raise ValueError(f"Invalid key in LoRA file: {k}, pls check the lora file format")
    if is_unet_format:
        print("Converting LoRA file to AI Toolkit format...")
        state_dict = convert_sd_scripts_to_ai_toolkit(state_dict)
    return state_dict

@torch.inference_mode()
def generate_image(pipeline, output_dir, prompt, height, width, guidance_scale, 
                   num_inference_steps, seed, batch_size, batches, 
                   lora_path1, lora_path2, lora_path3, lora_scale1, lora_scale2, lora_scale3,
                    progress=gr.Progress()):
    if not prompt:
        raise gr.Error("Please enter a prompt!")
    try:
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        lora_paths, lora_scales = [lora_path1, lora_path2, lora_path3], [lora_scale1, lora_scale2, lora_scale3]
        # Filter out empty lora paths
        active_loras = [(path, scale) for path, scale in zip(lora_paths, lora_scales) if path]
        
        lora_names = []
        # Load all selected LoRAs
        for lora_path, lora_scale in active_loras:
            lora_state_dict = load_lora_file(lora_path)
            adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
            adapter_name = adapter_name.replace(".", "_") # module name cannot contain "."
            pipeline.load_lora_weights(lora_state_dict, adapter_name=adapter_name)
            lora_names.append(adapter_name)
            print("Using lora {} with scale {}".format(adapter_name, lora_scale))
        assert len(lora_names) == len(active_loras)
        if active_loras:
            pipeline.set_adapters(lora_names, [scale for _, scale in active_loras])
        
        # Get timestamp for this generation batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed_start": seed,
            "batch_size": batch_size,
            "batches": batches,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales
        }
        
        all_images = []
        progress_text = f"Generating {batch_size*batches} images..."
        
        for i in progress.tqdm(range(batches), desc=progress_text):
            current_seed = seed + i
            generated = pipeline(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                num_images_per_prompt=batch_size,
                joint_attention_kwargs={"scale": 1.0},
                generator=torch.Generator("cpu").manual_seed(current_seed)
            ).images
            
            # Process all images in the batch
            for j, image in enumerate(generated):
                # Save the image with unique seed for each image in batch
                batch_seed = current_seed + j
                image_filename = f"{timestamp}_seed{batch_seed}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                all_images.append(image)
            
            yield all_images, f"Generated {len(all_images)}/{batch_size*batches} images. Current seed: {current_seed}"
        
        # Save metadata
        metadata_filename = f"{timestamp}_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Unload all LoRAs
        if active_loras:
            pipeline = unload_lora(pipeline)
            
        return all_images, f"Complete! Seeds used: {seed} to {seed + (batch_size * batches) - 1}. Images saved in {output_dir}"
    except Exception as e:
        raise gr.Error(f"Error generating image: {str(e)}")

def sequential_generate(pipe, output_dir, prompt_gen, keywords, system_prompt, height, width, guidance_scale, 
                      num_inference_steps, seed, batch_size, batches,
                      lora_path1, lora_path2, lora_path3, lora_scale1, lora_scale2, lora_scale3,
                      progress=gr.Progress()):
    # First generate the prompt
    for reasoning, result in prompt_gen.generate(keywords, system_prompt):
        yield reasoning, result, None, None, None, "Generating prompt..."
    
    # Then extract the prompt
    for extract_reasonsing, extract_result in prompt_gen.extract(result):
        yield reasoning, result, extract_reasonsing, extract_result, None, "Extracting prompt..."
    
    # Finally generate the image
    for images, progress_text in generate_image(pipe, output_dir, extract_result, height, width, 
                                                guidance_scale, num_inference_steps, 
                                                seed, batch_size, batches, 
                      lora_path1, lora_path2, lora_path3, lora_scale1, lora_scale2, lora_scale3):
        yield reasoning, result, extract_reasonsing, extract_result, images, progress_text

def load_parameters_from_metadata(metadata):
    """Load generation parameters from metadata into UI controls."""
    if not metadata:
        return [gr.update()] * (10 + 6)  # Additional 6 updates for the 3 loras (3 paths and 3 scales)
    
    # Handle legacy metadata format that had single lora_path and lora_alpha
    lora_paths = metadata.get("lora_paths", ["", "", ""])
    lora_scales = metadata.get("lora_scales", [1.0, 1.0, 1.0])
    
    # Handle legacy format with single lora
    if "lora_path" in metadata and "lora_alpha" in metadata:
        lora_paths[0] = metadata["lora_path"]
        lora_scales[0] = metadata["lora_alpha"]
    
    # Fill in missing loras if needed
    while len(lora_paths) < 3:
        lora_paths.append("")
    while len(lora_scales) < 3:
        lora_scales.append(1.0)
    
    return [
        gr.update(value=metadata.get("prompt", "")),
        gr.update(value=metadata.get("height", 1024)),
        gr.update(value=metadata.get("width", 768)),
        gr.update(value=metadata.get("guidance_scale", 3.5)),
        gr.update(value=metadata.get("num_inference_steps", 16)),
        gr.update(value=metadata.get("batch_size", 1)), 
        gr.update(value=metadata.get("batches", 1)),
        # shall always use the seed_start which is saved for the first seed in the batch
        gr.update(value=metadata.get("seed_start", -1)),
        # First lora
        gr.update(value=lora_paths[0]),
        gr.update(value=lora_scales[0]),
        # Second lora
        gr.update(value=lora_paths[1]),
        gr.update(value=lora_scales[1]),
        # Third lora
        gr.update(value=lora_paths[2]),
        gr.update(value=lora_scales[2])
    ]

def get_system_prompts_dir():
    """Get the system prompts directory path."""
    return Path(__file__).parents[2] / "presets" / "system_prompts"

def get_available_system_prompts():
    """Get list of available system prompt preset files."""
    prompts_dir = get_system_prompts_dir()
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return [f.name for f in prompts_dir.glob("*.txt")]

def load_system_prompt_file(filename):
    """Load a system prompt from a preset file."""
    prompts_dir = get_system_prompts_dir()
    try:
        with open(prompts_dir / filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"System prompt preset '{prompts_dir/filename}' not found")

def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_dev_model_4bit()
    if args.no_cpu_offload:
        pipe = pipe.to('cuda')
    else:
        pipe.enable_model_cpu_offload()

    prompt_gen = PromptGenerator()
    # Load default system prompt
    try:
        default_system_msg = load_system_prompt_file("default.txt")
        prompt_gen.system_msg = default_system_msg
    except Exception as e:
        print(f"Error loading default system prompt from: {get_system_prompts_dir()}, error:{str(e)}")
        print(f"Will use hard coded default system: {prompt_gen.system_msg}")

    def update_prompt_gen_preset(preset):
        """Update the prompt_gen object with selected preset."""
        prompt_gen.preset = preset
        return [
            gr.update(value=prompt_gen.api_base_url),
            gr.update(value=prompt_gen.api_key),
            gr.update(value=prompt_gen.model)
        ]

    def update_prompt_gen_settings(api_base_url_val, api_key_val, model_val):
        """Update the prompt_gen object with new API settings."""
        prompt_gen.api_base_url = str(api_base_url_val).strip()
        prompt_gen.api_key = api_key_val
        prompt_gen.model = str(model_val).strip() # in case there are leading/trailing spaces
    
    def load_system_prompt(filename):
        """Load the selected system prompt and update the UI."""
        if not filename:
            return gr.update()
        content = load_system_prompt_file(filename)
        prompt_gen.system_msg = content  # Update the prompt generator
        return gr.update(value=content)


    with gr.Blocks() as demo:
        gr.Markdown("# FLUX Image Generator")

        with gr.Tabs() as tabs:
            # First tab - Image Generation
            with gr.Tab("Image Generation"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            use_ai_prompt = gr.Checkbox(label="Use AI Prompt Generation", value=True)
                            with gr.Group() as ai_prompt_button_group:
                                keywords = gr.Textbox(label="Keywords for AI", 
                                                      placeholder="Enter keywords for AI prompt generation...", visible=True)
                                with gr.Row():
                                    generate_prompt_btn = gr.Button("AI:Generate Prompt", visible=True, variant="primary")
                                    extract_prompt_btn = gr.Button("AI:Extract Prompt", visible=True, variant='secondary')
                                    translate_btn = gr.Button("AI:Translate Prompt", visible=True)
                                all_in_one_btn = gr.Button("🎯 AI:One-Click Generate", variant="primary")
                            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")

                        with gr.Group():
                            with gr.Row():
                                height = gr.Slider(minimum=256, maximum=1024, step=64, value=1024, label="Height")
                                width = gr.Slider(minimum=256, maximum=1024, step=64, value=768, label="Width")
                            with gr.Row():
                                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=3.5, label="Guidance Scale")
                                num_inference_steps = gr.Slider(minimum=1, maximum=50, step=1, value=16, label="Number of Inference Steps")
                            with gr.Row():
                                batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Batch size")
                                batches = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Number of batches")
                            with gr.Row():
                                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                                refresh_lora = gr.Button("🔄", scale=1)
                            
                            # Replace single LoRA with multiple LoRAs
                            with gr.Group():
                                gr.Markdown("### LoRA Models (up to 3)")
                                
                                # LoRA 1
                                with gr.Row():
                                    lora_path1 = gr.Dropdown(
                                        choices=get_available_loras(),
                                        label="LoRA 1",
                                        value="",
                                        allow_custom_value=True,
                                        scale=10
                                    )
                                    lora_scale1 = gr.Slider(
                                        minimum=0, maximum=2, step=0.05, 
                                        value=1.0, 
                                        label="Scale", 
                                        scale=3
                                    )
                                
                                # LoRA 2
                                with gr.Row():
                                    lora_path2 = gr.Dropdown(
                                        choices=get_available_loras(),
                                        label="LoRA 2",
                                        value="",
                                        allow_custom_value=True,
                                        scale=10
                                    )
                                    lora_scale2 = gr.Slider(
                                        minimum=0, maximum=2, step=0.05, 
                                        value=1.0, 
                                        label="Scale", 
                                        scale=3
                                    )
                                
                                # LoRA 3
                                with gr.Row():
                                    lora_path3 = gr.Dropdown(
                                        choices=get_available_loras(),
                                        label="LoRA 3",
                                        value="",
                                        allow_custom_value=True,
                                        scale=10
                                    )
                                    lora_scale3 = gr.Slider(
                                        minimum=0, maximum=2, step=0.05, 
                                        value=1.0, 
                                        label="Scale", 
                                        scale=3
                                    )
                                
                            with gr.Row():
                                generate_btn = gr.Button("Generate Images")

                    with gr.Column():
                        output_gallery = gr.Gallery(label="Generated Images", columns=2, rows=4, every=1, preview=True)
                        seed_text = gr.Textbox(label="Generation Progress")

                        with gr.Group() as ai_prompt_info_group:
                            with gr.Row():
                                preset_radio = gr.Radio(
                                    choices=list(prompt_gen.PRESET_CONFIGS.keys()),
                                    value="cloud",
                                    label="API Preset (Experimental, local ollama may not work)",
                                    interactive=True
                                )
                            with gr.Row():
                                api_base_url = gr.Textbox(label="OpenAI API Base URL", 
                                                          value=prompt_gen.api_base_url, visible=True, interactive=True)
                                api_key = gr.Textbox(label="OpenAI API Key",
                                                     value=prompt_gen.api_key, visible=True, type="password", interactive=True)
                                model = gr.Textbox(label="Model", value=prompt_gen.model, visible=True, interactive=True)
                            with gr.Row():
                                system_prompt_preset = gr.Dropdown(
                                    choices=get_available_system_prompts(),
                                    value="default.txt",
                                    label="System Prompt Preset",
                                    interactive=True
                                )
                                refresh_presets = gr.Button("🔄", size='sm')
                            system_prompt = gr.Textbox(label="AI:System Prompt", 
                                                       value=prompt_gen.system_msg, visible=True, 
                                                       max_lines=8, lines=8, show_copy_button=True)
                            with gr.Row():
                                ai_think_process = gr.Textbox(
                                    label="AI:Thinking Process",
                                    placeholder="AI thinking will appear here...",
                                    every=1,
                                    interactive=False,
                                    visible=True,
                                    max_lines=8,
                                    lines=8,
                                    show_copy_button=True
                                )
                                ai_prompt = gr.Textbox(label="AI:Generated Prompt", 
                                                       placeholder="AI generated prompt will appear here...", 
                                                       every=1, interactive=True, visible=True,
                                                       max_lines=8, lines=8, show_copy_button=True)
                            with gr.Group():
                                temp_thinking_process = gr.Textbox(
                                    label="AI:Temp Thinking Process",
                                    placeholder="Temp thinking process to stage things...",
                                    every=1,
                                    interactive=False,
                                    visible=True,
                                    max_lines=4,
                                    lines=4,
                                    show_copy_button=True
                                )

            # Second tab - Image Gallery
            with gr.Tab("Generated Images Gallery"):
                with gr.Row():
                    gallery_refresh_btn = gr.Button("🔄 Refresh Gallery")
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Previous Generations",
                            columns=4,
                            rows=5,
                            allow_preview=False,
                            selected_index=None,
                        )
                        with gr.Row():
                            load_params_btn = gr.Button("📥 Load Parameters", variant="secondary")
                            delete_image_btn = gr.Button("🗑️ Delete Image", variant="stop")
                        image_metadata = gr.JSON(label="Image Metadata")
                    selected_image = gr.Image(label="Selected Image", type="pil")

        def toggle_prompt_inputs(use_ai):
            return {
                ai_prompt_button_group: gr.update(visible=use_ai),
                ai_prompt_info_group: gr.update(visible=use_ai)
            }

        use_ai_prompt.change(
            fn=toggle_prompt_inputs,
            inputs=[use_ai_prompt],
            outputs=[ai_prompt_button_group, ai_prompt_info_group]
        )

        # buttons associated with prompt generation
        generate_prompt_btn.click(
            fn= prompt_gen.generate,
            inputs=[keywords, system_prompt],
            outputs=[ai_think_process, ai_prompt],
        )
        extract_prompt_btn.click(
            fn=prompt_gen.extract,
            inputs=[ai_prompt],
            outputs=[temp_thinking_process, prompt],
        )
        translate_btn.click(
            fn=prompt_gen.translate,
            inputs=[ai_prompt],
            outputs=[temp_thinking_process, prompt],
        )

        # buttons associated with image generation
        generate_btn.click(
            fn=partial(generate_image, pipe, output_dir),
            inputs=[prompt, height, width, guidance_scale, num_inference_steps, seed, batch_size, batches, 
                    lora_path1, lora_path2, lora_path3, lora_scale1, lora_scale2, lora_scale3],
            outputs=[output_gallery, seed_text],
            show_progress='minimal',
            show_progress_on=seed_text
        )
        # button which does all the steps in one stop, 
        # prompt generation from keywords, extraction prompts and image generation
        all_in_one_btn.click(
            fn=partial(sequential_generate, pipe, output_dir, prompt_gen),
            inputs=[keywords, system_prompt, height, width, guidance_scale, 
                   num_inference_steps, seed, batch_size, batches, 
                   lora_path1, lora_path2, lora_path3, lora_scale1, lora_scale2, lora_scale3],
            outputs=[ai_think_process, ai_prompt, temp_thinking_process, prompt, output_gallery, seed_text],
            show_progress='minimal',
            show_progress_on=seed_text
        )

        # Change the refresh handler to use gr.update
        refresh_lora.click(
            fn=refresh_lora_choices,
            outputs=[lora_path1, lora_path2, lora_path3]
        )

        # Add event handlers for API settings changes
        api_base_url.change(
            fn=update_prompt_gen_settings,
            inputs=[api_base_url, api_key, model],
            outputs=None
        )
        api_key.change(
            fn=update_prompt_gen_settings,
            inputs=[api_base_url, api_key, model],
            outputs=None
        )
        model.change(
            fn=update_prompt_gen_settings,
            inputs=[api_base_url, api_key, model],
            outputs=None
        )

        # Add preset radio button handler
        preset_radio.change(
            fn=update_prompt_gen_preset,
            inputs=[preset_radio],
            outputs=[api_base_url, api_key, model]
        )

        # Add handlers for system prompt preset selection
        system_prompt_preset.change(
            fn=load_system_prompt,
            inputs=[system_prompt_preset],
            outputs=[system_prompt]
        )

        def refresh_system_prompts():
            return gr.update(choices=get_available_system_prompts())

        refresh_presets.click(
            fn=refresh_system_prompts,
            outputs=[system_prompt_preset]
        )

        all_images, all_image_paths = None, None
        def update_gallery():
            nonlocal all_images, all_image_paths
            all_images, all_image_paths= load_output_images(output_dir)
            return gr.update(value=all_images)

        def show_selected(evt: gr.SelectData, gallery_images):
            if not gallery_images:
                return None, None
            # Get the selected image path
            if evt.index >= len(all_image_paths):
                return None, None
            
            selected_path = all_image_paths[evt.index]
            base_name = os.path.splitext(os.path.basename(selected_path))[0]
            seed = int(base_name.split('_seed')[1])
            metadata = get_image_metadata(selected_path, output_dir)
            if metadata is not None:
                assert isinstance(metadata, dict)
                metadata['seed'] = seed
            return all_images[evt.index], metadata

        gallery_refresh_btn.click(
            fn=update_gallery,
            outputs=[gallery]
        )

        # Add selection event handler
        gallery.select(
            fn=show_selected,
            inputs=[gallery],
            outputs=[selected_image, image_metadata]
        )

        # Wire up the load parameters button
        load_params_btn.click(
            fn=load_parameters_from_metadata,
            inputs=[image_metadata],
            outputs=[
                prompt, height, width, guidance_scale,
                num_inference_steps, batch_size, batches, seed,
                lora_path1, lora_scale1,
                lora_path2, lora_scale2,
                lora_path3, lora_scale3
            ]
        )

        # Update the initial gallery on launch
        demo.load(
            fn=update_gallery,
            outputs=[gallery]
        )

        # Keep track of the currently selected image path
        current_selected_image_path, current_selected_image_idx = gr.State(None), gr.State(None)

        def update_selected_image_path(evt: gr.SelectData, gallery_images):
            if not gallery_images:
                return None
            if evt.index >= len(all_image_paths):
                return None
            return all_image_paths[evt.index], evt.index

        # Update the selected image path on selection
        gallery.select(
            fn=update_selected_image_path,
            inputs=[gallery],
            outputs=[current_selected_image_path, current_selected_image_idx]
        )

        def delete_selected_image(image_path, image_idx):
            success, message = delete_image(image_path, output_dir)
            nonlocal all_images, all_image_paths
            del all_images[image_idx]
            del all_image_paths[image_idx]
            if success:
                return gr.update(value=all_images), gr.update(value=None), gr.update(value=None), message
            return gr.update(), gr.update(), gr.update(), message

        delete_image_btn.click(
            fn=delete_selected_image,
            inputs=[current_selected_image_path, current_selected_image_idx],
            outputs=[gallery, selected_image, image_metadata, seed_text]
        )

    demo.launch(server_name="0.0.0.0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output", help="Output directory for generated images")
    # local experiment
    #   offloading BS1:1024x768:No-lora will consume peak GPU memory 7.6GB, generation time is 44s/image for 16 steps
    #   No offloading: BS1:1024x768:No-lora will consume peak GPU memory 11.9GB, generation time is 41s/image for 16 steps
    parser.add_argument("--no-cpu-offload", action="store_true", help="Enable CPU offload for model to save memory")
    return parser.parse_args()

if __name__ == "__main__":
    main()
