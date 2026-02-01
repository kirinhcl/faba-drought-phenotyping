#!/usr/bin/env python3
"""Extract vision foundation model features for all plant images.

Processes 11,626 plant images through DINOv2, CLIP, or BioCLIP backbones
and saves 768-dim CLS token embeddings to HDF5.

Usage:
    python scripts/extract_features.py --backbone dinov2
    python scripts/extract_features.py --backbone clip --batch_size 128
    python scripts/extract_features.py --backbone bioclip --sanity_check
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_and_convert_image(path: str) -> Image.Image:
    """Load image and convert RGBA to RGB by compositing on black background."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode == 'RGB':
        return img
    else:
        return img.convert('RGB')


def build_image_manifest(metadata_csv: str) -> list[dict[str, Any]]:
    df = pd.read_csv(metadata_csv)
    manifest = []
    
    for _, row in df.iterrows():
        plant_id = row['plant_id']
        side_dir = Path(row['image_side_dir'])
        top_dir = Path(row['image_top_dir'])
        timepoints = json.loads(row['available_timepoints'])
        
        for round_num in timepoints:
            for angle in ['000', '120', '240']:
                pattern = f"124-{round_num}-{plant_id}-RGB1-{angle}-FishEyeMasked.png"
                side_path = side_dir / angle / pattern
                if side_path.exists():
                    manifest.append({
                        'plant_id': plant_id,
                        'round': round_num,
                        'view_key': f'side_{angle}',
                        'path': str(side_path),
                        'treatment': row['treatment']
                    })
            
            pattern = f"124-{round_num}-{plant_id}-RGB2-FishEyeMasked.png"
            top_path = top_dir / pattern
            if top_path.exists():
                manifest.append({
                    'plant_id': plant_id,
                    'round': round_num,
                    'view_key': 'top',
                    'path': str(top_path),
                    'treatment': row['treatment']
                })
    
    return manifest


class PlantImageDataset(Dataset):
    def __init__(self, manifest: list[dict[str, Any]], transform):
        self.manifest = manifest
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.manifest[idx]
        img = load_and_convert_image(record['path'])
        img_tensor = self.transform(img)
        return img_tensor, record


class DINOv2Backbone(nn.Module):
    """DINOv2-B/14 wrapper that extracts 768-dim CLS + 256x768 patch tokens."""
    
    def __init__(self, device: str):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.model.eval()
        self.device = device
    
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    
    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cls_features, patch_features) both as numpy arrays."""
        batch = batch.to(self.device)
        outputs = self.model(pixel_values=batch)
        hidden_states = outputs.last_hidden_state
        cls_tokens = hidden_states[:, 0, :].cpu().numpy()
        patch_tokens = hidden_states[:, 1:, :].cpu().numpy()
        return cls_tokens, patch_tokens


class CLIPBackbone(nn.Module):
    """OpenAI CLIP ViT-B/16 wrapper that extracts 768-dim CLS token."""
    
    def __init__(self, device: str):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
        self.model.eval()
        self.device = device
    
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    
    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> tuple[np.ndarray, None]:
        """Returns (cls_features, None) as numpy array."""
        batch = batch.to(self.device)
        vision_outputs = self.model.vision_model(pixel_values=batch)
        cls_tokens = vision_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_tokens, None


class BioCLIPBackbone(nn.Module):
    """BioCLIP wrapper that extracts 768-dim CLS token via forward hook."""
    
    def __init__(self, device: str):
        super().__init__()
        import open_clip
        
        self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip'
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.cls_features = None
        
        # Register hook on the final LayerNorm to capture 768-dim CLS before projection
        def hook_fn(module, input, output):
            self.cls_features = output[:, 0, :].detach().cpu()
        
        # open_clip VisionTransformer: ln_post is the final LayerNorm before proj
        self.hook_handle = self.model.visual.ln_post.register_forward_hook(hook_fn)
    
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(img)
    
    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> tuple[np.ndarray, None]:
        """Returns (cls_features, None) as numpy array."""
        batch = batch.to(self.device)
        _ = self.model.encode_image(batch)
        cls_tokens = self.cls_features.numpy()
        self.cls_features = None
        return cls_tokens, None
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


def create_backbone(name: str, device: str) -> nn.Module:
    if name == 'dinov2':
        return DINOv2Backbone(device)
    elif name == 'clip':
        return CLIPBackbone(device)
    elif name == 'bioclip':
        return BioCLIPBackbone(device)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def get_processed_plants(h5_path: Path) -> set[str]:
    if not h5_path.exists():
        return set()
    
    processed = set()
    with h5py.File(h5_path, 'r') as f:
        processed = set(f.keys())
    return processed


def process_images(
    backbone_name: str,
    manifest: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: str
) -> None:
    processed_plants = get_processed_plants(output_path)
    if processed_plants:
        print(f"Resuming: {len(processed_plants)} plants already processed, skipping...")
        manifest = [m for m in manifest if m['plant_id'] not in processed_plants]
    
    if not manifest:
        print("All images already processed!")
        return
    
    print(f"Processing {len(manifest)} images...")
    
    backbone = create_backbone(backbone_name, device)
    
    dataset = PlantImageDataset(manifest, transform=backbone.preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_path, 'a')
    
    total_processed = 0
    current_plant = None
    start_time = time.time()
    
    try:
        for batch_imgs, batch_meta in tqdm(dataloader, desc=f"[{backbone_name}]"):
            cls_features, patch_features = backbone.extract(batch_imgs)

            for i in range(cls_features.shape[0]):
                plant_id = batch_meta['plant_id'][i]
                round_num = int(batch_meta['round'][i])
                view_key = batch_meta['view_key'][i]
                round_str = str(round_num)

                if plant_id not in h5_file:
                    h5_file.create_group(plant_id)
                if round_str not in h5_file[plant_id]:
                    h5_file[plant_id].create_group(round_str)

                group = h5_file[plant_id][round_str]

                if view_key in group:
                    continue

                group.create_dataset(view_key, data=cls_features[i], dtype='float32')

                # DINOv2 patch tokens: (256, 768) for spatial attention analysis
                if patch_features is not None:
                    group.create_dataset(
                        f"{view_key}_patches",
                        data=patch_features[i],
                        dtype='float32'
                    )

                total_processed += 1

                if current_plant != plant_id:
                    h5_file.flush()
                    current_plant = plant_id

                if total_processed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    pct = 100.0 * total_processed / len(manifest)
                    print(f"[{backbone_name}] {total_processed}/{len(manifest)} ({pct:.1f}%) — {rate:.0f} imgs/s")
    
    finally:
        h5_file.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({total_processed/elapsed:.1f} imgs/s)")
    print(f"Features saved to: {output_path}")


def run_sanity_check(
    backbone_name: str,
    metadata_csv: str,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str
) -> None:
    df = pd.read_csv(metadata_csv)
    
    np.random.seed(42)
    whc80 = df[df['treatment'] == 'WHC-80'].sample(5)
    whc30 = df[df['treatment'] == 'WHC-30'].sample(5)
    selected = pd.concat([whc80, whc30])
    
    selected_plants = set(selected['plant_id'].tolist())
    print(f"Sanity check: selected {len(selected_plants)} plants (5 WHC-80 + 5 WHC-30)")
    
    full_manifest = build_image_manifest(metadata_csv)
    manifest = [m for m in full_manifest if m['plant_id'] in selected_plants]
    
    print(f"Processing {len(manifest)} images for sanity check...")
    
    temp_h5 = output_dir / f"{backbone_name}_sanity.h5"
    temp_h5.unlink(missing_ok=True)
    
    process_images(
        backbone_name=backbone_name,
        manifest=manifest,
        output_path=temp_h5,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    
    plant_features = []
    plant_labels = []
    
    with h5py.File(temp_h5, 'r') as f:
        for plant_id in selected_plants:
            if plant_id not in f:
                continue
            
            features = []
            for round_key in f[plant_id].keys():
                for view_key in f[plant_id][round_key].keys():
                    if not view_key.endswith('_patches'):
                        features.append(f[plant_id][round_key][view_key][:])
            
            plant_avg = np.mean(features, axis=0)
            plant_features.append(plant_avg)
            
            treatment = selected[selected['plant_id'] == plant_id]['treatment'].iloc[0]
            plant_labels.append(0 if treatment == 'WHC-80' else 1)
    
    plant_features = np.array(plant_features)
    plant_labels = np.array(plant_labels)
    
    from sklearn.metrics import silhouette_score
    score = silhouette_score(plant_features, plant_labels)
    
    print(f"\nSilhouette score ({backbone_name}): {score:.4f}")
    
    sanity_path = output_dir / 'sanity_check.json'
    results = {}
    if sanity_path.exists():
        with open(sanity_path) as f:
            results = json.load(f)
    
    results[backbone_name] = {
        'silhouette_score': float(score),
        'num_plants': len(selected_plants),
        'num_images': len(manifest)
    }
    
    with open(sanity_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results appended to: {sanity_path}")
    
    temp_h5.unlink()


def main():
    parser = argparse.ArgumentParser(description="Extract vision foundation model features")
    parser.add_argument(
        '--backbone',
        type=str,
        required=True,
        choices=['dinov2', 'clip', 'bioclip'],
        help='Backbone model to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HDF5 path (default: features/{backbone}_features.h5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--sanity_check',
        action='store_true',
        help='Run sanity check on 10 plants only'
    )
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    metadata_csv = 'data/plant_metadata.csv'
    output_dir = Path('features')
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"{args.backbone}_features.h5"
    
    if args.sanity_check:
        run_sanity_check(
            backbone_name=args.backbone,
            metadata_csv=metadata_csv,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device
        )
    else:
        manifest = build_image_manifest(metadata_csv)
        print(f"Built manifest: {len(manifest)} total images")
        
        process_images(
            backbone_name=args.backbone,
            manifest=manifest,
            output_path=output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device
        )


if __name__ == '__main__':
    main()
