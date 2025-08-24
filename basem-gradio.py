import sys
import importlib.metadata
import os
from datetime import datetime
import random
import json
from PIL import PngImagePlugin, Image
import copy
import threading
import time
from collections import deque
import requests
from urllib.parse import unquote, urlparse
import traceback
import warnings
import re
import zipfile
import shutil

# --- Part 1: Final, Corrected, and Reliable Dependency Checker ---
print("--- Running Pre-flight Dependency Check ---")
print(f"--- Using Python executable: {sys.executable} ---")

packages_to_check = [
    "torch", "diffusers", "transformers", "accelerate", "safetensors", "gradio",
    "peft", "requests", "numpy", "opencv-python-headless",
    "controlnet_aux", "realesrgan", "gfpgan"
]

missing_packages = []
for package_name in packages_to_check:
    try:
        importlib.metadata.version(package_name)
        print(f"[✓] Found: {package_name}")
    except importlib.metadata.PackageNotFoundError:
        print(f"[X] Missing: {package_name}")
        missing_packages.append(package_name)

if missing_packages:
    print("\n[!] ERROR: Required libraries are missing for the current Python interpreter.")
    print("    To install all of them, please copy and run this single command in your terminal:")
    install_command = ' '.join(sorted(list(set(missing_packages))))
    print(f"\n    pip install {install_command}\n")
    print("    After the installation finishes, please re-run this script.")
    sys.exit()

warnings.filterwarnings("ignore")

import torch
import gradio as gr
import numpy as np
from safetensors import safe_open
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline,
    StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline,
    ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
    LMSDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler, PNDMScheduler
)
from controlnet_aux import CannyDetector
from huggingface_hub import hf_hub_download

FLUX_SUPPORTED = False
try:
    from diffusers import FluxPipeline
    FLUX_SUPPORTED = True
    print("[✓] FLUX pipeline is supported.")
except ImportError:
    print("[!] NOTE: Your 'diffusers' version does not include the newest FLUX pipeline.")
    class FluxPipeline: pass

print("\n[SUCCESS] All dependencies are present. Proceeding to launch the application...")

# --- Part 2: Advanced Application Architecture ---

IS_GENERATING_LOCK = threading.Lock()
SUPPORTED_EXTENSIONS = [".safetensors", ".ckpt"]
home_dir = os.path.expanduser('~')
base_output_folder = os.path.join(home_dir, 'Documents', 'AI_Generated_Images')
default_models_dir = os.path.join(base_output_folder, 'models')
default_loras_dir = os.path.join(base_output_folder, 'loras')
images_output_folder = os.path.join(base_output_folder, 'Images')
os.makedirs(default_models_dir, exist_ok=True)
os.makedirs(default_loras_dir, exist_ok=True)
os.makedirs(images_output_folder, exist_ok=True)


SAMPLER_MAP = {
    "DPM++ 2M Karras": {"class": DPMSolverMultistepScheduler, "kwargs": {"use_karras_sigmas": True}},
    "DPM++ 2M SDE Karras": {"class": DPMSolverMultistepScheduler, "kwargs": {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}},
    "Euler": {"class": EulerDiscreteScheduler, "kwargs": {}}, "Euler a": {"class": EulerAncestralDiscreteScheduler, "kwargs": {}},
    "LMS Karras": {"class": LMSDiscreteScheduler, "kwargs": {"use_karras_sigmas": True}}, "UniPC": {"class": UniPCMultistepScheduler, "kwargs": {}},
    "DDIM": {"class": DDIMScheduler, "kwargs": {}}, "PNDM": {"class": PNDMScheduler, "kwargs": {}}
}

PIPELINE_MAP = {
    "SD 1.5": {"txt2img": StableDiffusionPipeline, "img2img": StableDiffusionImg2ImgPipeline, "inpaint": StableDiffusionInpaintPipeline, "controlnet": StableDiffusionControlNetPipeline},
    "SDXL": {"txt2img": StableDiffusionXLPipeline, "img2img": StableDiffusionXLImg2ImgPipeline, "inpaint": StableDiffusionXLInpaintPipeline, "controlnet": StableDiffusionXLControlNetPipeline}
}
if FLUX_SUPPORTED:
    PIPELINE_MAP["FLUX"] = {"txt2img": FluxPipeline}

CONTROLNET_PREPROCESSORS = {"Canny": {"model": CannyDetector(), "hf_id": "lllyasviel/sd-controlnet-canny"}}
JOB_QUEUE = deque()
CANCEL_EVENT = threading.Event()
APP_STATE_DICT = {"status": "Idle", "latest_image": None, "latest_processed_image": None, "last_gen_params": {}, "log_history": ""}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UILogger:
    def __init__(self, original_stdout, app_state_dict):
        self.original_stdout = original_stdout
        self.app_state_dict = app_state_dict

    def write(self, message):
        self.original_stdout.write(message)
        self.app_state_dict["log_history"] += message
        if len(self.app_state_dict["log_history"]) > 50000:
            self.app_state_dict["log_history"] = self.app_state_dict["log_history"][-50000:]

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        return self.original_stdout.isatty()

def get_weighted_embeddings(prompt: str, tokenizer, text_encoder, device, max_length=77):
    prompt_pattern = re.compile(r'\s*\(\s*([^:]+?)\s*:\s*([\d\.]*)\s*\)\s*|([^\(\)]+)', re.VERBOSE)
    text_chunks, weight_chunks = [], []
    remaining_prompt = prompt.strip()
    while remaining_prompt:
        match = prompt_pattern.search(remaining_prompt)
        if not match:
            if remaining_prompt: text_chunks.append(remaining_prompt); weight_chunks.append(1.0)
            break
        pre_match_text = remaining_prompt[:match.start()].strip()
        if pre_match_text:
            text_chunks.append(pre_match_text); weight_chunks.append(1.0)
        weighted_text, weight_str, unweighted_text = match.groups()
        if weighted_text is not None:
            text_chunks.append(weighted_text.strip())
            try: weight = float(weight_str.strip()) if weight_str and weight_str.strip() else 1.0
            except ValueError: weight = 1.0
            weight_chunks.append(weight)
        elif unweighted_text is not None:
            text_chunks.append(unweighted_text.strip()); weight_chunks.append(1.0)
        remaining_prompt = remaining_prompt[match.end():]
    full_text = " ".join(text_chunks)
    tokens = tokenizer(full_text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    token_ids = tokens.input_ids.to(device)
    token_weights = torch.ones_like(token_ids, dtype=torch.float32, device=device)
    current_token_pos = 1
    for text_chunk, weight in zip(text_chunks, weight_chunks):
        if not text_chunk: continue
        chunk_tokens = tokenizer(text_chunk, add_special_tokens=False)['input_ids']
        num_chunk_tokens = len(chunk_tokens)
        if current_token_pos + num_chunk_tokens <= max_length:
            token_weights[0, current_token_pos : current_token_pos + num_chunk_tokens] = weight
        current_token_pos += num_chunk_tokens
    with torch.no_grad():
        embeddings = text_encoder(token_ids, output_hidden_states=True)
        embeddings = embeddings.last_hidden_state
    weighted_embeddings = embeddings * token_weights.unsqueeze(-1)
    return weighted_embeddings

UPSCALER = None
FACE_CORRECTOR = None

def get_model_architecture(model_path):
    if not model_path or not model_path.endswith('.safetensors'): return "SD 1.5"
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f: keys = {k for k in f.keys()}
        if FLUX_SUPPORTED and any("transformer.pos_embed.proj.weight" in k for k in keys): return "FLUX"
        if any("conditioner.embedders.1.model.ln_final.weight" in k for k in keys) or any("text_encoder_2" in k for k in keys): return "SDXL"
        return "SD 1.5"
    except Exception as e: print(f"Error inspecting model file: {e}"); return "SD 1.5"

def post_process_image(image, upscale, face_correct, app_state_dict):
    global UPSCALER, FACE_CORRECTOR
    processed_image = image
    if upscale:
        if CANCEL_EVENT.is_set(): raise InterruptedError("Cancelled before upscaling.")
        app_state_dict["status"] = "Upscaling (may take a moment)..."
        print("[Post-Process] Upscaling image...")
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            if UPSCALER is None:
                print("[Post-Process] Initializing RealESRGAN for the first time...")
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_path = hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4plus.pth")
                UPSCALER = RealESRGANer(
                    scale=4, model_path=model_path, model=model, tile=0, tile_pad=10,
                    pre_pad=0, half=True, gpu_id=0 if DEVICE.type == "cuda" else None
                )
                print("[Post-Process] RealESRGAN Initialized.")
            img_np = np.array(processed_image.convert("RGB"))
            output, _ = UPSCALER.enhance(img_np, outscale=4)
            processed_image = Image.fromarray(output)
            print("[Post-Process] Upscaling complete.")
        except ImportError as e:
            if "functional_tensor" in str(e): print("\n[!] WARNING: Real-ESRGAN upscaling is disabled due to a dependency version mismatch.\n    Your 'torchvision' version is likely too new for the 'basicsr' library.\n    To fix this, run: pip install --force-reinstall torchvision==0.15.2\n")
            else: print(f"\n[!] WARNING: Could not import RealESRGAN. Upscaling is disabled. Error: {e}\n")
        except Exception as e: print(f"[!] ERROR during upscaling: {e}\n{traceback.format_exc()}")
    if face_correct:
        if CANCEL_EVENT.is_set(): raise InterruptedError("Cancelled before face correction.")
        app_state_dict["status"] = "Correcting faces..."
        print("[Post-Process] Correcting faces...")
        try:
            from gfpgan import GFPGANer
            if FACE_CORRECTOR is None:
                print("[Post-Process] Initializing GFPGAN for the first time...")
                model_path = hf_hub_download(repo_id="TencentARC/GFPGAN", filename="GFPGANv1.4.pth")
                FACE_CORRECTOR = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE)
                print("[Post-Process] GFPGAN Initialized.")
            img_np = np.array(processed_image.convert("RGB"))
            _, _, restored_img = FACE_CORRECTOR.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
            processed_image = Image.fromarray(restored_img)
            print("[Post-Process] Face correction complete.")
        except ImportError as e: print(f"\n[!] WARNING: Could not import GFPGAN. Face correction is disabled. Error: {e}\n")
        except Exception as e: print(f"[!] ERROR during face correction: {e}\n{traceback.format_exc()}")
    return processed_image

def generation_worker(app_state_dict, job_queue):
    pipe, controlnet_model = None, None
    current_model_name, current_workflow, current_controlnet_id = None, None, None
    while True:
        if not job_queue: time.sleep(0.1); continue
        with IS_GENERATING_LOCK:
            job_params = job_queue.popleft()
            CANCEL_EVENT.clear()
            def callback_on_step(step, timestep, latents):
                if CANCEL_EVENT.is_set(): raise InterruptedError("Generation cancelled by user.")
            def flux_callback_on_step(pipe, step_index, timestep, callback_kwargs):
                if CANCEL_EVENT.is_set(): raise InterruptedError("Generation cancelled by user.")
                return callback_kwargs
            try:
                p = job_params; status_msg = "Initializing job..."
                app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                if CANCEL_EVENT.is_set(): raise InterruptedError("Job cancelled before start.")
                if not p.get('model_name'):
                    status_msg = "Error: Model not selected in job."
                    app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                    continue
                model_path = os.path.join(p['models_dir'], p['model_name'])
                detected_arch = get_model_architecture(model_path)
                p['model_arch'] = detected_arch
                needs_reload = (current_model_name != p['model_name']) or (current_workflow != p['workflow']) or (p['workflow'] == 'controlnet' and current_controlnet_id != p['controlnet_id'])
                if needs_reload:
                    if CANCEL_EVENT.is_set(): raise InterruptedError("Job cancelled before model load.")
                    pipe = controlnet_model = None; torch.cuda.empty_cache()
                    status_msg = f"Loading [{detected_arch}] {p['workflow']} pipeline for {p['model_name']}..."
                    app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                    PipelineClass = PIPELINE_MAP[detected_arch][p['workflow']]
                    load_kwargs = {"torch_dtype": torch.float16, "use_safetensors": True, "variant": "fp16"}
                    if p['workflow'] == 'controlnet':
                        status_msg = f"Loading ControlNet: {p['controlnet_id']}"
                        app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                        controlnet_model = ControlNetModel.from_pretrained(p['controlnet_id'], torch_dtype=torch.float16)
                        load_kwargs['controlnet'] = controlnet_model
                        current_controlnet_id = p['controlnet_id']
                    pipe = PipelineClass.from_single_file(model_path, **load_kwargs); pipe.to(DEVICE.type)
                    current_model_name, current_workflow = p['model_name'], p['workflow']
                if detected_arch == "SD 1.5": pipe.text_encoder.config.num_hidden_layers = 12 - int(p['clip_skip'])
                pipe.scheduler = SAMPLER_MAP[p['sampler_name']]["class"].from_config(pipe.scheduler.config, **SAMPLER_MAP[p['sampler_name']]["kwargs"])
                try: pipe.unload_lora_weights()
                except Exception: pass
                if p['lora_name'] and p['lora_name'] != "(None)":
                    status_msg = f"Loading LoRA: {os.path.basename(p['lora_name'])}"
                    app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                    pipe.load_lora_weights(os.path.join(p['loras_dir'], p['lora_name']))
                if int(p['seed']) == -1: p['seed'] = random.randint(0, 2**32 - 1)
                generator = torch.Generator(DEVICE.type).manual_seed(int(p['seed']))
                status_msg = f"Generating with seed {p['seed']}..."; app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                status_msg = "Conditioning prompts..."; app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                
                pipe_kwargs = {
                    "num_inference_steps": int(p['steps']), "guidance_scale": p['cfg_scale'],
                    "width": int(p['width']), "height": int(p['height']),
                    "generator": generator, "safety_checker": None
                }
                
                if detected_arch == "SDXL":
                    with torch.no_grad():
                        def get_weighted_text_embeddings(prompt: str, tokenizer, text_encoder, device):
                            max_length = tokenizer.model_max_length
                            prompt_pattern = re.compile(r'\s*\(\s*([^:]+?)\s*:\s*([\d\.]*)\s*\)\s*|([^\(\)]+)', re.VERBOSE)
                            text_chunks, weight_chunks = [], []
                            remaining_prompt = prompt.strip()
                            while remaining_prompt:
                                match = prompt_pattern.search(remaining_prompt)
                                if not match:
                                    if remaining_prompt: text_chunks.append(remaining_prompt); weight_chunks.append(1.0)
                                    break
                                pre_match_text = remaining_prompt[:match.start()].strip()
                                if pre_match_text: text_chunks.append(pre_match_text); weight_chunks.append(1.0)
                                weighted_text, weight_str, unweighted_text = match.groups()
                                if weighted_text is not None:
                                    text_chunks.append(weighted_text.strip())
                                    try: weight = float(weight_str.strip()) if weight_str and weight_str.strip() else 1.0
                                    except ValueError: weight = 1.0
                                    weight_chunks.append(weight)
                                elif unweighted_text is not None:
                                    text_chunks.append(unweighted_text.strip()); weight_chunks.append(1.0)
                                remaining_prompt = remaining_prompt[match.end():]
                            
                            full_text = " ".join(text_chunks)
                            tokens = tokenizer(full_text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
                            token_ids = tokens.input_ids.to(device)
                            token_weights = torch.ones_like(token_ids, dtype=torch.float32, device=device)
                            current_token_pos = 1
                            for text_chunk, weight in zip(text_chunks, weight_chunks):
                                if not text_chunk: continue
                                chunk_tokens = tokenizer(text_chunk, add_special_tokens=False)['input_ids']
                                num_chunk_tokens = len(chunk_tokens)
                                if current_token_pos + num_chunk_tokens <= max_length:
                                    token_weights[0, current_token_pos : current_token_pos + num_chunk_tokens] = weight
                                current_token_pos += num_chunk_tokens
                            
                            text_encoder_output = text_encoder(token_ids, output_hidden_states=True)
                            unweighted_embeds = text_encoder_output.last_hidden_state
                            weighted_embeds = unweighted_embeds * token_weights.unsqueeze(-1)
                            
                            return weighted_embeds, text_encoder_output.text_embeds

                        prompt_embeds_1, _ = get_weighted_text_embeddings(p['prompt'], pipe.tokenizer, pipe.text_encoder, DEVICE)
                        prompt_embeds_2, pooled_embeds = get_weighted_text_embeddings(p['prompt'], pipe.tokenizer_2, pipe.text_encoder_2)
                        neg_prompt_embeds_1, _ = get_weighted_text_embeddings(p['negative_prompt'], pipe.tokenizer, pipe.text_encoder, DEVICE)
                        neg_prompt_embeds_2, neg_pooled_embeds = get_weighted_text_embeddings(p['negative_prompt'], pipe.tokenizer_2, pipe.text_encoder_2)
                        final_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                        final_neg_prompt_embeds = torch.cat([neg_prompt_embeds_1, neg_prompt_embeds_2], dim=-1)
                        
                        pipe_kwargs.update({
                            "prompt_embeds": final_prompt_embeds,
                            "pooled_prompt_embeds": pooled_embeds,
                            "negative_prompt_embeds": final_neg_prompt_embeds,
                            "negative_pooled_prompt_embeds": neg_pooled_embeds
                        })
                elif detected_arch == "SD 1.5":
                    prompt_embeds = get_weighted_embeddings(p['prompt'], pipe.tokenizer, pipe.text_encoder, DEVICE, max_length=pipe.tokenizer.model_max_length)
                    neg_prompt_embeds = get_weighted_embeddings(p['negative_prompt'], pipe.tokenizer, pipe.text_encoder, DEVICE, max_length=pipe.tokenizer.model_max_length)
                    pipe_kwargs.update({"prompt_embeds": prompt_embeds, "negative_prompt_embeds": neg_prompt_embeds})
                else:
                    pipe_kwargs.update({"prompt": p['prompt'], "negative_prompt": p['negative_prompt']})
                
                if p['lora_name'] and p['lora_name'] != "(None)": pipe_kwargs["cross_attention_kwargs"] = {"scale": p['lora_weight']}
                if p['workflow'] == 'img2img':
                    pipe_kwargs.update({"image": p['init_image'].convert("RGB"), "strength": p['denoising_strength']})
                elif p['workflow'] == 'inpaint':
                    pipe_kwargs.update({"image": p['init_image'].convert("RGB"), "mask_image": p['mask_image'], "strength": p['denoising_strength']})
                elif p['workflow'] == 'controlnet':
                    status_msg = f"Preprocessing with {p['controlnet_preprocessor']}"
                    app_state_dict["status"] = status_msg; print(f"[Worker] {status_msg}")
                    preprocessor = CONTROLNET_PREPROCESSORS[p['controlnet_preprocessor']]['model']
                    control_image = preprocessor(p['init_image']); pipe_kwargs['image'] = control_image
                elif p['workflow'] != 'txt2img':
                    raise ValueError(f"Unknown workflow type received: {p['workflow']}")
                
                if detected_arch == "FLUX":
                    pipe_kwargs["callback_on_step_end"] = flux_callback_on_step
                else:
                    pipe_kwargs["callback"] = callback_on_step
                    pipe_kwargs["callback_steps"] = 1

                if CANCEL_EVENT.is_set(): raise InterruptedError("Job cancelled before generation.")
                image = pipe(**pipe_kwargs).images[0]
                processed_image = post_process_image(image, p['upscale'], p['face_correct'], app_state_dict)
                workflow_params = {k: v for k, v in p.items() if not isinstance(v, Image.Image)}
                metadata = PngImagePlugin.PngInfo(); metadata.add_text("parameters", json.dumps(workflow_params, default=str))
                file_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed-{p['seed']}.png"
                processed_image.save(os.path.join(images_output_folder, file_name), pnginfo=metadata)
                status_msg = f"Done! Seed: {p['seed']}"
                app_state_dict.update({"latest_image": image, "latest_processed_image": processed_image, "last_gen_params": copy.deepcopy(p), "status": status_msg})
                print(f"[Worker] {status_msg}")
            except InterruptedError as e:
                print(f"\n--- Job Cancelled ---\n{e}\n------------------------")
                app_state_dict["status"] = f"Job Cancelled: {e}"
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n--- Generation Error ---\n{traceback.format_exc()}\n------------------------")
                app_state_dict["status"] = f"Error: {e}"
            finally:
                current_status = app_state_dict["status"]
                is_error = "Error" in current_status
                is_cancelled = "Cancel" in current_status
                if not job_queue and not is_error and not is_cancelled:
                    app_state_dict["status"] = "Idle"
                    print("[Worker] Idle. Waiting for jobs...")
                CANCEL_EVENT.clear()

def scan_files(folder_path, extensions):
    try: return [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in extensions)]
    except FileNotFoundError: return []

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def download_file(url, dest_folder):
    if not url: return "Error: URL is empty."
    try:
        if "drive.google.com" in url:
            file_id = None
            if "open?id=" in url:
                file_id = url.split("open?id=")[1].split("&")[0]
            elif "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]

            if not file_id:
                return "Error: Could not extract file ID from Google Drive URL."

            filename = f"downloaded_gdrive_file.safetensors"
            filepath = os.path.join(dest_folder, filename)
            print(f"Downloading Google Drive file with ID {file_id} to {filepath}")
            download_file_from_google_drive(file_id, filepath)
            return f"Success! Downloaded {filename} from Google Drive."

        else:
            os.makedirs(dest_folder, exist_ok=True); filename = unquote(os.path.basename(urlparse(url).path))
            if not any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS): return f"Error: Invalid file extension."
            filepath = os.path.join(dest_folder, filename); print(f"Downloading {url} to {filepath}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            return f"Success! Downloaded {filename}"
    except Exception as e: print(f"Download failed: {e}"); return f"Error: {e}"

def update_file_lists(models_dir, loras_dir):
    model_files = scan_files(models_dir, SUPPORTED_EXTENSIONS)
    lora_files = scan_files(loras_dir, SUPPORTED_EXTENSIONS)
    return gr.update(choices=model_files), gr.update(choices=["(None)"] + lora_files)

def handle_download(url, dest_folder, models_dir, loras_dir):
    status = download_file(url, dest_folder)
    model_choices, lora_choices = update_file_lists(models_dir, loras_dir)
    return status, model_choices, lora_choices

def load_workflow_from_image(image_obj):
    # This must return 15 values to match the output list
    num_outputs = 15
    if image_obj is None: return [gr.update()] * num_outputs
    try:
        params = json.loads(image_obj.info.get("parameters", "{}"))
        if not params: return [gr.update() for _ in range(num_outputs)]
    except Exception: return [gr.update() for _ in range(num_outputs)]
    model_arch = params.get("model_arch")
    width = params.get("width", 1024 if model_arch == "SDXL" else 512)
    height = params.get("height", 1024 if model_arch == "SDXL" else 768)
    models_dir = params.get("models_dir")
    loras_dir = params.get("loras_dir")
    if models_dir and loras_dir:
        model_files = scan_files(models_dir, SUPPORTED_EXTENSIONS)
        lora_files = scan_files(loras_dir, SUPPORTED_EXTENSIONS)
        model_dropdown_update = gr.update(choices=model_files, value=params.get("model_name"))
        lora_dropdown_update = gr.update(choices=["(None)"] + lora_files, value=params.get("lora_name"))
    else:
        model_dropdown_update = gr.update(value=params.get("model_name"))
        lora_dropdown_update = gr.update(value=params.get("lora_name"))
    return (
        gr.update(value=models_dir) if models_dir else gr.update(),
        gr.update(value=loras_dir) if loras_dir else gr.update(),
        model_dropdown_update, lora_dropdown_update,
        gr.update(value=params.get("lora_weight")), gr.update(value=params.get("clip_skip")),
        gr.update(value=params.get("prompt")), gr.update(value=params.get("negative_prompt")),
        gr.update(value=params.get("sampler_name")), gr.update(value=params.get("steps")),
        gr.update(value=params.get("cfg_scale")), gr.update(value=width),
        gr.update(value=height), gr.update(value=params.get("seed"))
    )

def ui_update_on_tick():
    queue_size = len(JOB_QUEUE)
    status_text = f"Queue: {queue_size} | {APP_STATE_DICT.get('status', 'Initializing...')}"
    latest_image = APP_STATE_DICT.get("latest_processed_image")
    log_history = APP_STATE_DICT.get("log_history", "")
    return status_text, latest_image, log_history

def cancel_all_jobs():
    if not JOB_QUEUE and not IS_GENERATING_LOCK.locked():
        print("Nothing to cancel. Queue is empty and worker is idle.")
        gr.Info("Nothing to cancel.")
        return
    print("[UI] Cancellation requested. Clearing queue and interrupting current job.")
    JOB_QUEUE.clear()
    CANCEL_EVENT.set()
    APP_STATE_DICT["status"] = "Cancellation requested..."
    gr.Info("Cancellation requested. The current job will stop shortly.")

# --- History Tab Functions (Corrected and with Full Debugging) ---

def get_history_filenames():
    if not os.path.exists(images_output_folder):
        return []
    files = os.listdir(images_output_folder)
    return sorted([f for f in files if f.lower().endswith('.png')], 
                  key=lambda f: os.path.getmtime(os.path.join(images_output_folder, f)), 
                  reverse=True)

def refresh_history_list():
    print("[DEBUG] Refreshing history file list.")
    return gr.update(choices=get_history_filenames(), value=[])

def create_download_zip(filenames_to_zip, zip_name="archive.zip"):
    print(f"\n--- [DEBUG] DOWNLOAD: Starting ZIP creation for '{zip_name}' ---")
    if not filenames_to_zip:
        print("[DEBUG] DOWNLOAD: No files were provided. Aborting ZIP creation.")
        gr.Warning("No files selected to download.")
        return None, None 

    zip_path = os.path.join(base_output_folder, zip_name)
    print(f"[DEBUG] DOWNLOAD: ZIP file will be created at: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            print(f"[DEBUG] DOWNLOAD: Opened zip file. Attempting to add {len(filenames_to_zip)} file(s)...")
            for filename in filenames_to_zip:
                full_path = os.path.join(images_output_folder, filename)
                if os.path.exists(full_path):
                    zipf.write(full_path, os.path.basename(filename))
                    print(f"  [+] Added: {filename}")
                else:
                    print(f"  [!] Skipped (not found): {filename}")
        
        print(f"[DEBUG] DOWNLOAD: ZIP creation successful.")
        status_update = f"Successfully created {zip_name} at {datetime.now().strftime('%H:%M:%S')}"
        print(f"[DEBUG] DOWNLOAD: CRITICAL STEP - Returning file path '{zip_path}' and status '{status_update}'")
        return zip_path, status_update
    except Exception as e:
        print(f"[ERROR] DOWNLOAD: Failed to create ZIP file: {e}\n{traceback.format_exc()}")
        gr.Error(f"Failed to create ZIP file: {e}")
        return None, f"Failed: {e}"

def download_all_history_images_func(dummy_input=None):
    print(f"[DEBUG] DOWNLOAD: 'Download All' button was clicked. (Received dummy input: {dummy_input})")
    return create_download_zip(get_history_filenames(), "history_archive.zip")

def download_selected_history_images_func(selected_filenames):
    print(f"[DEBUG] DOWNLOAD: 'Download Selected' button clicked with {len(selected_filenames)} files.")
    return create_download_zip(selected_filenames, "selected_images.zip")

def update_history_view(selected_filenames):
    single_view_visible = len(selected_filenames) == 1
    multi_view_visible = len(selected_filenames) > 0
    
    preview_image = None
    metadata = ""
    path_for_send = ""
    
    if single_view_visible:
        filename = selected_filenames[0]
        filepath = os.path.join(images_output_folder, filename)
        path_for_send = filepath
        preview_image = Image.open(filepath)
        try:
            params = json.loads(preview_image.info.get("parameters", "{}"))
            metadata = json.dumps(params, indent=2)
        except Exception as e:
            metadata = f"Could not read metadata: {e}"
            
    return (
        gr.update(visible=multi_view_visible),
        gr.update(visible=single_view_visible),
        preview_image,
        metadata,
        path_for_send
    )

def send_params_to_generator(filepath):
    print(f"[DEBUG] HISTORY: 'Load Parameters' button clicked for image: {filepath}")
    if not filepath or not os.path.exists(filepath):
        gr.Warning("File path is missing or file does not exist.")
        return [gr.update()] * 15

    try:
        img = Image.open(filepath)
        params = json.loads(img.info.get("parameters", "{}"))
        if not params:
            gr.Warning("No metadata found in image.")
            return [gr.update()] * 15
        print("[DEBUG] HISTORY: Successfully parsed metadata from image.")

        model_arch = params.get("model_arch")
        width = params.get("width", 1024 if model_arch == "SDXL" else 512)
        height = params.get("height", 1024 if model_arch == "SDXL" else 768)
        
        models_dir_val = params.get("models_dir", default_models_dir)
        loras_dir_val = params.get("loras_dir", default_loras_dir)
        model_choices = scan_files(models_dir_val, SUPPORTED_EXTENSIONS)
        lora_choices = ["(None)"] + scan_files(loras_dir_val, SUPPORTED_EXTENSIONS)
        print("[DEBUG] HISTORY: Updating Generator UI components now.")

        return (
            gr.update(value=models_dir_val),
            gr.update(value=loras_dir_val),
            gr.update(choices=model_choices, value=params.get("model_name")),
            gr.update(choices=lora_choices, value=params.get("lora_name")),
            gr.update(value=params.get("lora_weight")),
            gr.update(value=params.get("clip_skip")),
            gr.update(value=params.get("prompt")),
            gr.update(value=params.get("negative_prompt")),
            gr.update(value=params.get("sampler_name")),
            gr.update(value=params.get("steps")),
            gr.update(value=params.get("cfg_scale")),
            gr.update(value=width),
            gr.update(value=height),
            gr.update(value=params.get("seed")),
            gr.update(selected="generator_tab")
        )
    except Exception as e:
        gr.Warning(f"Could not load parameters: {e}")
        print(f"[ERROR] HISTORY: Failed to load parameters: {e}")
        return [gr.update()] * 15

def send_to_img2img(filepath):
    if not filepath: return gr.update(), gr.update()
    img = Image.open(filepath).convert("RGB")
    return img, gr.update(selected="img2img")

def send_to_inpaint(filepath):
    if not filepath: return gr.update(), gr.update()
    img = Image.open(filepath).convert("RGB")
    return {"background": img, "layers": [], "composite": None}, gr.update(selected="inpaint")

def send_to_controlnet(filepath):
    if not filepath: return gr.update(), gr.update()
    img = Image.open(filepath).convert("RGB")
    return img, gr.update(selected="controlnet")

css_style = """
#floating_generate_button { position: fixed !important; top: 25px !important; left: 25px !important; width: 120px !important; height: 120px !important; border-radius: 50% !important; z-index: 999 !important; font-size: 24px !important; line-height: 1.2 !important; border-width: 2px !important; }
.compact_list .gr-form {display: flex; flex-direction: column; height: 70vh; overflow-y: scroll;}
"""
js_code = """
() => {
    function adjustWeight(textarea, delta) {
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        if (start === end) return;
        let selectedText = textarea.value.substring(start, end);
        let newText;
        const weightRegex = /\\(\\s*(.*?)\\s*:\\s*([\\d\\.]+)\\s*\\)/;
        const match = selectedText.match(weightRegex);
        if (match) {
            let text = match[1];
            let weight = parseFloat(match[2]);
            let newWeight = Math.max(0, weight + delta).toFixed(2);
            newText = `(${text}:${newWeight})`;
        } else {
            let newWeight = (1.0 + delta).toFixed(2);
            newText = `(${selectedText}:${newWeight})`;
        }
        textarea.value = textarea.value.substring(0, start) + newText + textarea.value.substring(end);
        textarea.selectionStart = start;
        textarea.selectionEnd = start + newText.length;
        const inputEvent = new Event('input', { bubbles: true });
        textarea.dispatchEvent(inputEvent);
    }
    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            const generateButton = document.getElementById('main_generate_button');
            if (generateButton) { generateButton.click(); }
        }
        const activeElement = document.activeElement;
        if (event.ctrlKey && (activeElement.id === 'prompt_input' || activeElement.id === 'neg_prompt_input')) {
            if (event.key === 'ArrowUp') {
                event.preventDefault();
                adjustWeight(activeElement, 0.1);
            } else if (event.key === 'ArrowDown') {
                event.preventDefault();
                adjustWeight(activeElement, -0.1);
            }
        }
    });
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css_style, js=js_code) as demo:
    floating_generate_button = gr.Button("Generate", elem_id="floating_generate_button")
    gr.Markdown("# Local Multi-Workflow Image Generator")
    
    with gr.Tabs() as top_level_tabs:
        with gr.Tab("Generator", id="generator_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Load Workflow from Image", open=False):
                        workflow_uploader = gr.Image(label="Upload PNG with workflow data", type="pil")
                    with gr.Accordion("Downloader", open=True):
                        gr.Markdown("⚠️ *Only download files from sources you trust. Google Drive links are supported.*")
                        download_status = gr.Textbox(label="Download Status", interactive=False)
                        with gr.Row():
                            model_url_input, download_model_button = gr.Textbox(label="Model Download URL", scale=4), gr.Button("Download", scale=1)
                        with gr.Row():
                            lora_url_input, download_lora_button = gr.Textbox(label="LoRA Download URL", scale=4), gr.Button("Download", scale=1)
                    gr.Markdown("### Setup")
                    models_dir_input = gr.Textbox(label="Models Folder", value=default_models_dir)
                    loras_dir_input = gr.Textbox(label="LoRAs Folder", value=default_loras_dir)
                    refresh_button = gr.Button("Refresh File Lists")
                    gr.Markdown("### Selection & Weights")
                    model_dropdown = gr.Dropdown(label="Model", choices=scan_files(default_models_dir, SUPPORTED_EXTENSIONS), allow_custom_value=True)
                    lora_dropdown = gr.Dropdown(label="LoRA (Optional)", choices=["(None)"] + scan_files(default_loras_dir, SUPPORTED_EXTENSIONS), value="(None)", allow_custom_value=True)
                    lora_weight_slider = gr.Slider(label="LoRA Strength", minimum=0.0, maximum=1.5, value=1.0, step=0.05)
                    gr.Markdown("### Prompts")
                    prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter prompt here. Use (word:1.2) syntax for emphasis.", elem_id="prompt_input")
                    neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, elem_id="neg_prompt_input")
                    active_workflow_state = gr.Textbox("txt2img", visible=False)
                    with gr.Tabs() as main_tabs:
                        with gr.Tab("Text-to-Image", id="txt2img"): pass
                        with gr.Tab("Image-to-Image", id="img2img"):
                            i2i_input_image = gr.Image(label="Input Image", type="pil", sources=['upload'])
                            i2i_denoising_strength = gr.Slider(label="Denoising Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.01)
                        with gr.Tab("Inpainting", id="inpaint"):
                            inpaint_input_image = gr.ImageEditor(label="Draw Mask on Image", type="pil")
                            inpaint_denoising_strength = gr.Slider(label="Denoising Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.01)
                        with gr.Tab("ControlNet", id="controlnet"):
                            cn_input_image = gr.Image(label="Input Image for ControlNet", type="pil", sources=['upload'])
                            cn_preprocessor = gr.Dropdown(label="Preprocessor", choices=list(CONTROLNET_PREPROCESSORS.keys()), value="Canny")
                    def on_tab_select(evt: gr.SelectData):
                        label_to_key = { "Text-to-Image": "txt2img", "Image-to-Image": "img2img", "Inpainting": "inpaint", "ControlNet": "controlnet" }
                        return label_to_key.get(evt.value, "txt2img")
                    main_tabs.select(on_tab_select, None, active_workflow_state)
                    gr.Markdown("### Generation Parameters")
                    sampler_dropdown = gr.Dropdown(label="Sampler", choices=list(SAMPLER_MAP.keys()), value="DPM++ 2M Karras")
                    steps_slider = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, value=7.0, step=0.5)
                    clip_skip_slider = gr.Slider(label="CLIP Skip (SD1.5 Only)", minimum=0, maximum=4, value=0, step=1)
                    with gr.Row():
                        width_slider = gr.Slider(label="Width", minimum=256, maximum=2048, value=512, step=64)
                        height_slider = gr.Slider(label="Height", minimum=256, maximum=2048, value=768, step=64)
                    seed_input = gr.Number(label="Seed", value=-1, precision=0)
                    with gr.Accordion("Post-Processing (Optional)", open=False):
                        upscale_toggle = gr.Checkbox(label="Upscale")
                        face_correct_toggle = gr.Checkbox(label="Face Correction")
                    
                    queue_multiplier_input = gr.Number(label="Batch Count", value=1, minimum=1, precision=0, step=1)
                    
                    with gr.Row():
                        generate_button = gr.Button("Generate", variant="primary", elem_id="main_generate_button", scale=3)
                        cancel_button = gr.Button("Cancel All", variant="stop", scale=1)
                with gr.Column(scale=2):
                    status_output = gr.Textbox(label="Live Status", interactive=False, lines=1)
                    image_output = gr.Image(label="Generated Image", type="pil", show_label=False, height=768)
                    log_output = gr.Textbox(label="Console Log", lines=10, interactive=False, max_lines=20)

        with gr.Tab("History", id="history_tab"):
            history_image_path_state = gr.State("")
            dummy_download_trigger = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    history_refresh_button = gr.Button("Refresh History")
                    download_all_button = gr.DownloadButton("Download All as ZIP")
                    download_selected_button = gr.DownloadButton("Download Selected as ZIP", visible=False)
                    
                    history_file_list = gr.CheckboxGroup(label="Image History", choices=get_history_filenames(), elem_classes="compact_list")

                with gr.Column(scale=2):
                    history_preview = gr.Image(label="Preview", type="pil", show_label=False, height=512)
                    with gr.Group(visible=False) as history_image_controls:
                        send_to_generate_button = gr.Button("Load Parameters")
                        gr.Markdown("#### Send to...")
                        send_to_img2img_button = gr.Button("Image-to-Image")
                        send_to_inpaint_button = gr.Button("Inpaint")
                        send_to_controlnet_button = gr.Button("ControlNet")
                        history_metadata_output = gr.Textbox(label="Image Metadata", lines=10, interactive=False)
    
    def add_to_queue_from_ui(
        queue_multiplier,
        active_workflow, i2i_img, i2i_strength, inpaint_data, inpaint_strength, cn_img, cn_prep,
        models_dir, loras_dir, model_name, lora_name, lora_weight, clip_skip,
        prompt, negative_prompt, sampler_name, steps, cfg_scale, width, height, seed,
        upscale, face_correct
    ):
        if not model_name:
            gr.Warning("Please select a model first!")
            return

        base_job_params = {
            'models_dir': models_dir, 'loras_dir': loras_dir, 'model_name': model_name,
            'lora_name': lora_name, 'lora_weight': lora_weight, 'clip_skip': clip_skip,
            'prompt': prompt, 'negative_prompt': negative_prompt, 'sampler_name': sampler_name,
            'steps': steps, 'cfg_scale': cfg_scale, 'width': width, 'height': height,
            'seed': seed, 'upscale': upscale, 'face_correct': face_correct, 'workflow': active_workflow
        }

        if active_workflow == 'img2img':
            if not i2i_img: gr.Warning("Please provide an input image for Image-to-Image."); return
            base_job_params.update({'init_image': i2i_img, 'denoising_strength': i2i_strength})

        elif active_workflow == 'inpaint':
            try:
                if (isinstance(inpaint_data, dict) and inpaint_data.get('background') and inpaint_data.get('layers')):
                    init_img = inpaint_data['background']
                    mask_rgba = inpaint_data['layers'][0]
                    final_mask = mask_rgba.split()[-1]
                    base_job_params.update({'init_image': init_img.convert("RGB"), 'mask_image': final_mask, 'denoising_strength': inpaint_strength})
                else:
                    gr.Warning("Missing image or mask. Please upload an image AND draw a mask before generating."); return
            except Exception as e:
                print(f"[Error] Inpaint data extraction failed unexpectedly: {e}\n{traceback.format_exc()}")
                gr.Warning("Inpainting failed. Check the console log for technical details."); return

        elif active_workflow == 'controlnet':
            if not cn_img: gr.Warning("Please provide an input image for ControlNet."); return
            base_job_params.update({ 'init_image': cn_img, 'controlnet_preprocessor': cn_prep, 'controlnet_id': CONTROLNET_PREPROCESSORS[cn_prep]['hf_id'] })
        
        batch_count = max(1, int(queue_multiplier or 1))

        if batch_count == 1 and int(seed) != -1:
            comparable_params = {k: v for k, v in base_job_params.items() if not isinstance(v, Image.Image)}
            last_params = {k: v for k, v in APP_STATE_DICT.get("last_gen_params", {}).items() if not isinstance(v, Image.Image)}
            if comparable_params == last_params:
                gr.Info(f"Parameters unchanged for seed {seed}. Displaying previous image."); return

        jobs_to_add = []
        for _ in range(batch_count):
            job_copy = copy.deepcopy(base_job_params)
            if batch_count > 1: job_copy['seed'] = -1
            jobs_to_add.append(job_copy)
        
        JOB_QUEUE.extend(jobs_to_add)
        gr.Info(f"Added {len(jobs_to_add)} job(s) to the queue.")

    common_inputs = [
        models_dir_input, loras_dir_input, model_dropdown, lora_dropdown, lora_weight_slider,
        clip_skip_slider, prompt_input, neg_prompt_input, sampler_dropdown, steps_slider,
        cfg_slider, width_slider, height_slider, seed_input, upscale_toggle, face_correct_toggle
    ]
    all_ui_inputs = [
        queue_multiplier_input, active_workflow_state, i2i_input_image, i2i_denoising_strength,
        inpaint_input_image, inpaint_denoising_strength, cn_input_image, cn_preprocessor
    ] + common_inputs
    
    generate_button.click(fn=add_to_queue_from_ui, inputs=all_ui_inputs, outputs=None)
    floating_generate_button.click(fn=add_to_queue_from_ui, inputs=all_ui_inputs, outputs=None)
    cancel_button.click(fn=cancel_all_jobs, inputs=None, outputs=None)
    refresh_button.click(fn=update_file_lists, inputs=[models_dir_input, loras_dir_input], outputs=[model_dropdown, lora_dropdown])
    download_model_button.click(fn=handle_download, inputs=[model_url_input, models_dir_input, models_dir_input, loras_dir_input], outputs=[download_status, model_dropdown, lora_dropdown])
    download_lora_button.click(fn=handle_download, inputs=[lora_url_input, loras_dir_input, models_dir_input, loras_dir_input], outputs=[download_status, model_dropdown, lora_dropdown])
    
    # Corrected: The output list for this function must have 14 items
    workflow_outputs = [models_dir_input, loras_dir_input, model_dropdown, lora_dropdown, lora_weight_slider, clip_skip_slider, prompt_input, neg_prompt_input, sampler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input]
    workflow_uploader.upload(fn=load_workflow_from_image, inputs=[workflow_uploader], outputs=workflow_outputs)

    # --- History Tab Event Handlers (Corrected and Stabilized) ---
    history_refresh_button.click(fn=refresh_history_list, inputs=None, outputs=[history_file_list])
    
    download_all_button.click(
        fn=download_all_history_images_func, 
        inputs=[dummy_download_trigger], 
        outputs=[download_all_button, dummy_download_trigger]
    )
    download_selected_button.click(
        fn=download_selected_history_images_func, 
        inputs=[history_file_list], 
        outputs=[download_selected_button, dummy_download_trigger]
    )
    
    history_file_list.change(
        fn=update_history_view,
        inputs=[history_file_list],
        outputs=[download_selected_button, history_image_controls, history_preview, history_metadata_output, history_image_path_state]
    )
    
    # Corrected: This list MUST have 15 components to match the 15 return values from send_params_to_generator
    generator_param_outputs = [
        models_dir_input, loras_dir_input,
        model_dropdown, lora_dropdown, lora_weight_slider, clip_skip_slider,
        prompt_input, neg_prompt_input, sampler_dropdown, steps_slider,
        cfg_slider, width_slider, height_slider, seed_input, top_level_tabs
    ]
    send_to_generate_button.click(fn=send_params_to_generator, inputs=[history_image_path_state], outputs=generator_param_outputs)
    send_to_img2img_button.click(fn=send_to_img2img, inputs=[history_image_path_state], outputs=[i2i_input_image, main_tabs])
    send_to_inpaint_button.click(fn=send_to_inpaint, inputs=[history_image_path_state], outputs=[inpaint_input_image, main_tabs])
    send_to_controlnet_button.click(fn=send_to_controlnet, inputs=[history_image_path_state], outputs=[cn_input_image, main_tabs])

    timer = gr.Timer(1)
    timer.tick(fn=ui_update_on_tick, inputs=None, outputs=[status_output, image_output, log_output])

if __name__ == "__main__":
    original_stdout = sys.stdout
    sys.stdout = UILogger(original_stdout, APP_STATE_DICT)
    worker_thread = threading.Thread(target=generation_worker, args=(APP_STATE_DICT, JOB_QUEUE), daemon=True)
    worker_thread.start()
    demo.launch(share=False, inline=False, allowed_paths=[images_output_folder, base_output_folder])