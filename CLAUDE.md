# claude.md

## Response Structure Format

All responses will follow this consistent structure:

1. **English Description / Response**: Clear explanation of concepts, tasks, or answers in plain language without technical jargon.

2. **Step-by-Step Outline**: For coding tasks, a structured breakdown of implementation steps without actual code.

3. **Context**: Relevant file references for each step, specifying which components (classes, functions, constants) are important.

## Project Overview

### English Description
The A3 model is an advanced video generation architecture that extends the Wan2.1 model with multi-reference conditioning capabilities. Rather than conditioning on just a single reference image, A3 incorporates triple conditioning using object references, mask/placement references, and background references. Our goal is to implement an efficient fine-tuning method using LoRA (Low-Rank Adaptation) that targets only the specific layers responsible for this multi-reference conditioning process. This approach will reduce trainable parameters to less than 0.1% of the full model, making fine-tuning significantly more efficient while maintaining quality. 

### Step-by-Step Outline
1. Understand A3's conditioning mechanisms from inference code
2. Identify target modules for LoRA adaptation
3. Create training pipeline that mimics inference conditioning flow
4. Implement configuration-driven approach using TOML files
5. Design efficient data loading focused on triple references

### Context
- **./infer.py**: Reference for understanding how A3 processes multiple conditioning inputs
- **./training/configs/a3_lora.toml**: Defines configuration for LoRA adaptation, including target modules
- **./training/configs/a3_dataset.toml.toml**: Defines dataset structure and processing requirements
- **./models/**: Contains model architecture definitions and configurations

## Architecture Analysis

### English Description
The A3 model extends Wan2.1 by adding specialized attention mechanisms that handle multiple reference images. These references influence generation through dedicated projection layers in the cross-attention blocks. Specifically, the architecture adds `add_k_proj` and `add_v_proj` layers that process object/mask and background references respectively. The LoRA adaptation targets precisely these layers, allowing the model to learn new reference-to-video associations without modifying the base model weights. The inference code reveals a dual-pathway processing of references through both CLIP (for high-level features) and VAE (for spatial details).

The model consists of several key components:

1. **A2Model** (transformer): A transformer-based architecture specifically designed for video generation, featuring:
   - 40 transformer layers with 40 attention heads
   - 5120-dimensional hidden states (40 heads × 128 head dimension)
   - 13824-dimensional feedforward networks
   - Specialized WanAttnProcessor2_0 attention processor that handles multiple conditioning inputs
   - Support for LoRA adaptation with targeted parameter efficiency

2. **CLIP Vision Model**: A ViT-H/14 vision transformer (1280-dimensional) that processes reference images at 224×224 resolution but supports interpolated position encoding for 512×512 images.

3. **UMT5 Text Encoder**: A large text encoder (4096-dimensional) with 24 layers and 64 attention heads for processing textual prompts.

4. **AutoencoderKLWan**: A specialized VAE that:
   - Works with 5D tensors (batch, channels, time, height, width)
   - Has a z-dimension of 16 channels
   - Uses temporal downsampling to efficiently process video frames
   - Includes specialized latent normalization (latents_mean and latents_std)

5. **UniPCMultistepScheduler**: A flow-prediction based scheduler optimized for video generation.

### Step-by-Step Outline
1. Identify reference encoding pathways in the model
2. Map attention mechanisms responsible for conditioning
3. Analyze how reference features are processed and integrated
4. Determine which specific sublayers handle different reference types
5. Confirm target modules from configuration match architecture


## Technical Details

### English Description
The A3 model utilizes distinct processing techniques for different reference types. Each reference image (object, mask, background) undergoes specialized preparation to optimize its conditioning effect. Object images maintain aspect ratio with white background padding, while mask and background images are precisely resized to 512×512. During training, conditional dropout can be applied to masks and backgrounds to improve generalization. The model components follow a structured directory organization, and each component requires specific tensor dimension formatting, especially the VAE's 5D requirements and CLIP vision model's 4D expectations.

### Step-by-Step Outline
1. Implement reference processing with type-specific handling
2. Apply conditional dropout during training with configurable probabilities
3. Set up model directory structure with all required components
4. Format tensors with correct dimensions for each model component
5. Ensure proper memory formatting and precision for efficient processing

### Context

**A3 Conditioning Differences**
* **Object Images**: Placed on white background then resized with padding to maintain aspect ratio (no cropping)
* **Mask Images**: Resized with padding to exactly 512×512 to maintain aspect ratio
* **Background Images**: Resized with padding to exactly 512×512 to maintain aspect ratio
* **Conditional Dropout**: During training, masks and backgrounds can be randomly dropped with configurable probabilities (typically 0.3)

**Model Directory Structure**
```
/path/to/model/
├── image_encoder/             # Contains CLIP vision model files
│   └── config.json            # CLIP configuration
├── image_processor/           # Contains CLIP image processor
│   └── preprocessor_config.json  # Image processor configuration
├── vae/                       # Contains VAE model
│   └── config.json            # VAE configuration 
├── tokenizer/                 # Contains UMT5 tokenizer
│   ├── tokenizer_config.json  # Tokenizer configuration
│   └── special_tokens_map.json  # Special tokens
├── text_encoder/              # Contains UMT5 text encoder
│   └── config.json            # Text encoder configuration
├── transformer/               # Contains A2 transformer
│   └── config.json            # Transformer configuration
└── model_index.json           # Overall model index
```

**Tensor Dimensions and Formats**
* **VAE (AutoencoderKLWan)**:
   * Input: 5D tensor in format [B, C, T, H, W] where:
      * B: Batch size
      * C: Channels (typically 3 for RGB images)
      * T: Time dimension (for single images/frames, set T=1 using unsqueeze(2))
      * H/W: Height/Width of images (must be divisible by 16)
   * For single images, convert [B, C, H, W] → [B, C, 1, H, W] before processing
   * Memory format must be set to contiguous: `tensor.to(memory_format=torch.contiguous_format)`
   * Use bfloat16 for memory efficiency: `.to(device, dtype=torch.bfloat16)`
   * Uses a spatial scale factor of 8: latent dimensions = input dimensions / 8
* **CLIP Vision Model**:
   * Input: 4D tensor in format [B, C, H, W]
   * No time dimension needed (unlike the VAE)
   * Standard image processing applies
   * Native training size is 224×224, but can handle different sizes using `interpolate_pos_encoding=True`
   * For A3, use interpolation parameter when processing 512×512 images to avoid resizing

### Context
- **./models/**: All model configs are here. 

## Directory Structure

### English Description
Our custom extension of A3 with modified conditioning. It adds a canny mask showing object placement. Includes background image (same as A2 but preprocessed differently). 

### Step-by-Step Outline
```
/path/to/dataset/
  ├── 001.mp4                  # Target video
  ├── 001.txt                  # Caption/prompt
  ├── 001_object.png           # Object reference (required)
  ├── 001_mask.png             # Placement mask (optional)
  ├── 001_background.png       # Background image (optional)
  ├── 002.mp4
  └── ...
```

### Context
- **./training/configs/a3_dataset.toml.toml**: Referenced by main config for dataset settings

### Context
- **infer.py**: Lines 30-75 show model component initialization and reference processing
- **./training/configs/a3_lora.toml**: Target modules section defines which layers receive LoRA adaptation
  ```
  target_modules = [
    "blocks.*.attn2.add_k_proj",  # Object and mask conditioning projection 
    "blocks.*.attn2.add_v_proj"   # Background conditioning projection
  ]
  ```
- **./models/**: Provides detailed layer definitions for analyzing architecture

## Configuration Management

### English Description
The A3 training process is driven by TOML configuration files that separate model definitions, training parameters, and dataset specifications. The core configuration file `./training/configs/a3_lora.toml` defines the entire training process, from model paths to LoRA parameters to optimization settings. This configuration-driven approach allows for flexible experimentation without code changes. The training script must parse these configurations rather than hardcoding values, enabling rapid iteration across different training scenarios.

### Step-by-Step Outline
1. Parse main configuration file for high-level parameters
2. Extract model configuration including checkpoints path
3. Load LoRA-specific settings (rank, target modules)
4. Configure optimizer based on parameters section
5. Set up dataset loading from referenced dataset config file

### Context
- **./training/configs/a3_lora.toml**: Primary configuration containing:
  - `output_dir`: Where training artifacts are stored
  - `dataset`: Reference to dataset configuration file
  - `[model]` section: Model configuration and paths
  - `[adapter]` section: LoRA-specific parameters
  - `[optimizer]` section: Optimization parameters
- **./training/configs/a3_dataset.toml.toml**: Referenced by main config for dataset settings

## Data Pipeline

### English Description
The A3 model requires a specialized dataset structure with multiple reference images per video. Each training sample must include a target video, textual prompt, and three reference images (object, thing/mask, background). The inference code reveals different processing requirements for each reference type - object and thing references need aspect-preserved padding while background references use center cropping. The data pipeline must replicate these exact processing steps to maintain consistency between training and inference.

### Step-by-Step Outline
1. Organize dataset according to expected directory structure
2. Implement reference image loading with correct processing per type
3. Set up video frame extraction with proper resolution and timing
4. Create batching mechanism that preserves reference associations
5. Apply identical preprocessing as used in inference pipeline

### Context
- **infer.py**: Lines 77-102 show reference image processing:
  ```python
  # Different processing for different reference types
  if image_id == 0 or image_id == 1:  # Object or thing references
      image_vae = _crop_and_resize_pad(image, height=height, width=width)
  else:  # Background reference
      image_vae = _crop_and_resize(image, height=height, width=width)
  ```
- **Utility functions**: `_crop_and_resize_pad()` and `_crop_and_resize()` show exact preprocessing requirements
- **./training/configs/a3_dataset.toml.toml**: Contains dataset path and structure information

## Model Initialization

### English Description
Initializing the A3 model for LoRA training requires loading several distinct components - the transformer backbone, VAE, image encoder, and text encoder. Each component loads from a specific subdirectory in the checkpoint path. After loading these components, LoRA adaptation must be selectively applied only to the target modules while freezing all other parameters. The inference code provides the exact component loading pattern, which must be replicated for training while adding the LoRA adaptation layer. Precision settings from the configuration (e.g., bfloat16) must be applied consistently.

### Step-by-Step Outline
1. Load model components from checkpoint directories
2. Apply correct precision settings from configuration
3. Set up LoRA adaptation on targeted attention projection layers
4. Freeze all parameters except LoRA weights
5. Configure model for training mode with proper settings

### Context
- **infer.py**: Lines 30-53 show model component loading:
  ```python
  image_encoder = CLIPVisionModel.from_pretrained(pipeline_path, subfolder="image_encoder", torch_dtype=torch.float32)
  vae = AutoencoderKLWan.from_pretrained(pipeline_path, subfolder="vae", torch_dtype=torch.float32)
  transformer = A3Model.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True)
  ```
- **./training/configs/a3_lora.toml**: Contains model precision settings and LoRA configuration:
  ```
  [model]
  dtype = 'bfloat16'
  transformer_dtype = 'float8'
  
  [adapter]
  type = 'lora'
  rank = 32
  dtype = 'bfloat16'
  ```
- **./models/**: Contains model architecture definitions needed during initialization