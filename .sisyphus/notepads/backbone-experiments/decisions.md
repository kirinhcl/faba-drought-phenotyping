## [2026-02-10] BioCLIP 2 Feature Extraction Approach

### Decision
Use `model.encode_image()` for BioCLIP 2 instead of the `ln_post` hook pattern used by BioCLIPBackbone.

### Rationale
- BioCLIP v1 (ViT-B/16): transformer width=768, ln_post hook → 768-dim ✓
- BioCLIP 2 (ViT-L/14): transformer width=1024, ln_post hook → 1024-dim ✗
- BioCLIP 2 CLIP projection: 1024→768, encode_image() → 768-dim ✓

### Trade-offs
**Pros:**
- Dimensionally compatible with existing pipeline (768-dim)
- No model architecture changes needed
- Simpler implementation

**Cons:**
- Semantic inconsistency: DINOv2/CLIP/BioCLIP use pre-projection features, BioCLIP 2 uses post-projection
- Post-projection features are in CLIP alignment space (trained for text-image matching)
- May affect fair comparison (though projection IS part of the learned representation)

### Mitigation
- Documented in code comments
- Should be mentioned in paper's methods section
- The projection layer is part of BioCLIP 2's architecture, so this is still a valid comparison

