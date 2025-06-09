import os
import torch
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from moviepy.editor import VideoFileClip
from einops import rearrange
from datetime import datetime
import gradio as gr
from PIL import Image

from src.dataset.test_preprocess import preprocess
from src.dataset.utils import save_videos_grid, seed_everything, get_head_exp_motion_bucketid
from src.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.pipelines.hunyuan_svd_pipeline import HunyuanLongSVDPipeline
from src.models.condition.unet_3d_svd_condition_ip import UNet3DConditionSVDModel, init_ip_adapters
from src.models.condition.coarse_motion import HeadExpression, HeadPose
from src.models.condition.refine_motion import IntensityAwareMotionRefiner
from src.models.condition.pose_guider import PoseGuider
from src.models.dinov2.models.vision_transformer import vit_large, ImageProjector

# --- Global Variables for Models and Config ---
# Load configuration (adjust path if needed)
CONFIG_PATH = "./config/hunyuan-portrait.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}. Please ensure it exists.")

cfg = OmegaConf.load(CONFIG_PATH)

# Create output directory if it doesn't exist
output_dir = f"{cfg.output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
print(f"Output directory set to: {output_dir}")

tmp_path = './tmp_path/'
os.makedirs(tmp_path,exist_ok=True)

# Initialize models and pipeline (this will run once when the script starts)
print("Initializing models... This might take a while.")

if cfg.seed is not None:
    seed_everything(cfg.seed)

# Determine weight dtype
if cfg.weight_dtype == "fp16":
    weight_dtype = torch.float16
elif cfg.weight_dtype == "fp32":
    weight_dtype = torch.float32
elif cfg.weight_dtype == "bf16":
    weight_dtype = torch.bfloat16
else:
    raise ValueError(f"Do not support weight dtype: {cfg.weight_dtype}")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: CUDA not available, running on CPU. This will be very slow.")
    # If on CPU, override weight_dtype to fp32 for broader compatibility,
    # as not all ops support fp16/bf16 on CPU.
    weight_dtype = torch.float32


vae = AutoencoderKLTemporalDecoder.from_pretrained(
    cfg.pretrained_model_name_or_path,
    subfolder="vae",
    variant="fp16" if weight_dtype == torch.float16 else None # Diffusers handles dtype better if None for fp32/bf16
).to(device, dtype=weight_dtype)

val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    cfg.pretrained_model_name_or_path,
    subfolder="scheduler"
)

unet = UNet3DConditionSVDModel.from_config(
    cfg.pretrained_model_name_or_path,
    subfolder="unet",
    variant="fp16" if weight_dtype == torch.float16 else None
)
init_ip_adapters(unet, cfg.num_adapter_embeds, cfg.ip_motion_scale)
# Load UNet checkpoint before moving to device and setting dtype
print(f"Loading UNet checkpoint from: {cfg.unet_checkpoint_path}")
unet.load_state_dict(torch.load(cfg.unet_checkpoint_path, map_location="cpu"), strict=True)
unet.to(device, dtype=weight_dtype)


pose_guider = PoseGuider(
    conditioning_embedding_channels=320,
    block_out_channels=(16, 32, 96, 256)
)
print(f"Loading PoseGuider checkpoint from: {cfg.pose_guider_checkpoint_path}")
pose_guider.load_state_dict(torch.load(cfg.pose_guider_checkpoint_path, map_location="cpu"), strict=True)
pose_guider.to(device, dtype=weight_dtype)

motion_expression_model = HeadExpression(cfg.input_expression_dim)
print(f"Loading MotionExpression checkpoint from: {cfg.motion_expression_checkpoint_path}")
motion_expression_model.load_state_dict(torch.load(cfg.motion_expression_checkpoint_path, map_location="cpu"), strict=True)
motion_expression_model.to(device) # dtype handling might be internal or needs specific attention

motion_headpose_model = HeadPose()
print(f"Loading MotionHeadpose checkpoint from: {cfg.motion_pose_checkpoint_path}")
motion_headpose_model.load_state_dict(torch.load(cfg.motion_pose_checkpoint_path, map_location="cpu"), strict=True)
motion_headpose_model.to(device)

motion_proj = IntensityAwareMotionRefiner(
    input_dim=cfg.input_expression_dim,
    output_dim=cfg.motion_expression_dim,
    num_queries=cfg.num_queries
)
print(f"Loading MotionProj checkpoint from: {cfg.motion_proj_checkpoint_path}")
motion_proj.load_state_dict(torch.load(cfg.motion_proj_checkpoint_path, map_location="cpu"), strict=True)
motion_proj.to(device) # dtype handling

image_encoder = vit_large(
    patch_size=14,
    num_register_tokens=4,
    img_size=526,
    init_values=1.0,
    block_chunks=0,
    backbone=True,
    layers_output=True,
    add_adapter_layer=[3, 7, 11, 15, 19, 23],
    visual_adapter_dim=384,
)
print(f"Loading DinoV2 checkpoint from: {cfg.dino_checkpoint_path}")
image_encoder.load_state_dict(torch.load(cfg.dino_checkpoint_path, map_location="cpu"), strict=True)
image_encoder.to(device, dtype=weight_dtype)

image_proj = ImageProjector(cfg.num_img_tokens, cfg.num_queries, dtype=unet.dtype)
print(f"Loading ImageProj checkpoint from: {cfg.image_proj_checkpoint_path}")
image_proj.load_weights(cfg.image_proj_checkpoint_path, strict=True)
image_proj.to(device, dtype=weight_dtype)

# Set models to evaluation mode
vae.eval()
unet.eval()
pose_guider.eval()
motion_expression_model.eval()
motion_headpose_model.eval()
motion_proj.eval()
image_encoder.eval()
image_proj.eval()

motion_expression_model.requires_grad_(False)
motion_headpose_model.requires_grad_(False)

pipe = HunyuanLongSVDPipeline(
    unet=unet,
    image_encoder=image_encoder,
    image_proj=image_proj,
    vae=vae,
    pose_guider=pose_guider,
    scheduler=val_noise_scheduler,
)

arcface_session = None
if cfg.use_arcface:
    print(f"Loading ArcFace model from: {cfg.arcface_model_path}")
    arcface_session = ort.InferenceSession(cfg.arcface_model_path, providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])

print("Models initialized successfully.")
# --- End of Global Initializations ---

@torch.no_grad()
def generate_video_from_image_and_video(image, video_path):
    """
    Generates a video based on a source image and a driving video.
    """
    print("Starting video generation process...")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(tmp_path, f"source_image_{timestamp}.png")
        Image.fromarray(image).save(image_path)
        print(f"Processing with image: {image_path} and video: {video_path}")

        save_video_path = os.path.join(output_dir, f'{timestamp}.mp4')
        print(f"Generating and writing to: {save_video_path}")

        # Preprocessing
        print("Starting preprocessing...")
        sample = preprocess(image_path, video_path, limit=cfg.frame_num,
                            image_size=cfg.arcface_img_size, area=cfg.area, det_path=cfg.det_path)
        print("Preprocessing finished.")

        ref_img = sample['ref_img'].unsqueeze(0).to(device, dtype=weight_dtype)
        transformed_images = sample['transformed_images'].unsqueeze(0).to(device, dtype=weight_dtype)
        arcface_img_data = sample['arcface_image'] # Renamed to avoid conflict
        lmk_list = sample['lmk_list']

        if not cfg.use_arcface or arcface_img_data is None or arcface_session is None:
            arcface_embeddings = np.zeros((1, cfg.arcface_img_size))
        else:
            arcface_img_data_np = arcface_img_data.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
            arcface_embeddings = arcface_session.run(None, {"data": arcface_img_data_np})[0]
            arcface_embeddings = arcface_embeddings / np.linalg.norm(arcface_embeddings)

        dwpose_images = sample['img_pose']
        motion_pose_images = sample['motion_pose_image']
        motion_face_images = sample['motion_face_image']

        pose_cond_tensor_all = []
        driven_feat_all = []
        uncond_driven_feat_all = []

        print("Extracting motion features...")
        batch_size = cfg.n_sample_frames
        num_total_frames = motion_pose_images.shape[0]

        for idx in range(0, num_total_frames, batch_size):
            motion_pose_batch = motion_pose_images[idx:idx+batch_size].to(device)
            motion_face_batch = motion_face_images[idx:idx+batch_size].to(device)
            pose_cond_batch = dwpose_images[idx:idx+batch_size].to(device, dtype=weight_dtype)
            lmks_batch = lmk_list[idx:idx+batch_size]
            
            motion_bucket_id_head, motion_bucket_id_exp = get_head_exp_motion_bucketid(lmks_batch)

            motion_feature = motion_expression_model(motion_face_batch)
            motion_bucket_id_head_tensor = torch.IntTensor([motion_bucket_id_head]).to(device)
            motion_bucket_id_exp_tensor = torch.IntTensor([motion_bucket_id_exp]).to(device)
            motion_feature_embed = motion_proj(motion_feature, motion_bucket_id_head_tensor, motion_bucket_id_exp_tensor)

            driven_pose_feat = motion_headpose_model(motion_pose_batch * 2 + 1)

            driven_pose_feat_embed = torch.cat([driven_pose_feat['rotation'], driven_pose_feat['translation'] * 0], dim=-1)
            
            driven_feat_batch = torch.cat([motion_feature_embed, driven_pose_feat_embed.unsqueeze(1).repeat(1, motion_feature_embed.shape[1], 1)], dim=-1)
            driven_feat_batch = driven_feat_batch.unsqueeze(0) # Add batch dim for pipeline
            uncond_driven_feat_batch = torch.zeros_like(driven_feat_batch)

            pose_cond_batch = pose_cond_batch.unsqueeze(0) # Add batch dim
            pose_cond_batch = rearrange(pose_cond_batch, 'b f c h w -> b c f h w')

            pose_cond_tensor_all.append(pose_cond_batch)
            driven_feat_all.append(driven_feat_batch)
            uncond_driven_feat_all.append(uncond_driven_feat_batch)

        # Concatenate batches
        pose_cond_tensor_full = torch.cat(pose_cond_tensor_all, dim=2) # Concatenate along frame dimension
        uncond_driven_feat_full = torch.cat(uncond_driven_feat_all, dim=1) # Concatenate along sequence/token dimension
        driven_feat_full = torch.cat(driven_feat_all, dim=1)

        current_num_frames = pose_cond_tensor_full.shape[2] # Number of actual frames processed
        
        pose_cond_tensor_padded_list = []
        driven_feat_padded_list = []
        uncond_driven_feat_padded_list = []

        if cfg.pad_frames > 0:
            # Pad start
            for i in range(cfg.pad_frames):
                weight = (i + 1) / (cfg.pad_frames +1) # Gradual weight, avoid 0 if it causes issues
                pose_cond_tensor_padded_list.append(pose_cond_tensor_full[:, :, :1])
                driven_feat_padded_list.append(driven_feat_full[:, :1] * weight)
                uncond_driven_feat_padded_list.append(uncond_driven_feat_full[:, :1]) # Uncond usually stays zero or placeholder

            pose_cond_tensor_padded_list.append(pose_cond_tensor_full)
            driven_feat_padded_list.append(driven_feat_full)
            uncond_driven_feat_padded_list.append(uncond_driven_feat_full)

            # Pad end
            for i in range(cfg.pad_frames):
                weight = (i + 1) / (cfg.pad_frames+1) # Gradual weight
                pose_cond_tensor_padded_list.append(pose_cond_tensor_full[:, :, -1:])
                driven_feat_padded_list.append(driven_feat_full[:, -1:] * (1 - weight))
                uncond_driven_feat_padded_list.append(uncond_driven_feat_full[:, :1])

            pose_cond_tensor_full = torch.cat(pose_cond_tensor_padded_list, dim=2)
            driven_feat_full = torch.cat(driven_feat_padded_list, dim=1)
            uncond_driven_feat_full = torch.cat(uncond_driven_feat_padded_list, dim=1)
            
            num_frames_for_pipe = current_num_frames + cfg.pad_frames * 2
        else:
            num_frames_for_pipe = current_num_frames
    except Exception as e:
        print("---!!! An error occurred in generate_video_from_image_and_video !!!---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        raise gr.Error(f"An internal error occurred: {e}")        

    print(f"Calling generation pipeline with {num_frames_for_pipe} target frames...")
    # Ensure dtypes of inputs to the pipe are correct
    video_output_frames = pipe(
        ref_img.clone(), # Ensure clone if modified
        transformed_images.clone(),
        pose_cond_tensor_full.to(device, dtype=weight_dtype),
        driven_feat_full.to(device, dtype=unet.dtype), # Match UNet's internal dtype for conditional embeddings
        uncond_driven_feat_full.to(device, dtype=unet.dtype),
        height=cfg.height,
        width=cfg.width,
        num_frames=num_frames_for_pipe, # This should be the total number of frames expected by the pipe
        decode_chunk_size=cfg.decode_chunk_size,
        motion_bucket_id=cfg.motion_bucket_id,
        fps=cfg.fps, # FPS for generation guidance if used by scheduler
        noise_aug_strength=cfg.noise_aug_strength,
        min_guidance_scale1=cfg.min_appearance_guidance_scale,
        max_guidance_scale1=cfg.max_appearance_guidance_scale,
        min_guidance_scale2=cfg.min_motion_guidance_scale,
        max_guidance_scale2=cfg.max_motion_guidance_scale,
        overlap=cfg.overlap,
        shift_offset=cfg.shift_offset,
        frames_per_batch=cfg.temporal_batch_size, # MODIFICATION: Use the new dedicated config parameter.
        num_inference_steps=cfg.num_inference_steps,
        i2i_noise_strength=cfg.i2i_noise_strength,
        arcface_embeddings=arcface_embeddings,
    ).frames

    # Post-processing video frames    
    print("Pipeline generation finished.")
    video_tensor = video_output_frames.cpu()
    video_tensor.mul_(0.5).add_(0.5).clamp_(0, 1)
    video_processed = video_tensor

    if cfg.pad_frames > 0:
        video_final = video_processed[:, :, cfg.pad_frames:-cfg.pad_frames]
    else:
        video_final = video_processed
    
    # Get FPS from original video for saving
    try:
        input_video_clip = VideoFileClip(video_path)
        output_fps = input_video_clip.fps
        input_video_clip.close()
    except Exception as e:
        print(f"Warning: Could not read FPS from input video: {e}. Using default cfg.fps: {cfg.fps}")
        output_fps = cfg.fps

    print(f"Saving video to {save_video_path} with FPS: {output_fps}")
    save_videos_grid(video_final, save_video_path, n_rows=1, fps=output_fps)
    print("Video saved.")

    return save_video_path

if __name__ == "__main__":
    # --- Gradio Interface Setup ---
    print("Starting Gradio app...")
    iface = gr.Interface(
        fn=generate_video_from_image_and_video,
        inputs=[
            gr.Image(label="Upload Image", height=400),
            gr.Video(label="Upload Video", height=400),
        ],
        outputs=gr.Video(label="Generated Video", height=400),
        title="HunyuanPortrait Animation",
        description="Upload a source image and a driving video to generate a new video where the image is animated by the video's motion.",
    )
    # To make it accessible on the network, use share=True
    # To run on a specific port, use server_port=XXXX
    iface.launch(
        server_name='0.0.0.0', 
        server_port=8089,
        share=False,
    )