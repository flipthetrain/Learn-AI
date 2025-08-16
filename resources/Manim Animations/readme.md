# Manim Animations

Programmatic animations of AI/ML concepts that don't have good existing visualisations.
Built with [Manim Community Edition](https://www.manim.community/).

## Animations

| File | Concept | What it shows |
|---|---|---|
| [lora_rank_decomposition.py](./lora_rank_decomposition.py) | LoRA | How A×B approximates a weight update at a fraction of the parameter cost |
| [rag_pipeline.py](./rag_pipeline.py) | RAG | The full index → embed → retrieve → generate flow |
| [tokenization_bpe.py](./tokenization_bpe.py) | Tokenization / BPE | How Byte-Pair Encoding merges characters into subword tokens |
| [moe_expert_routing.py](./moe_expert_routing.py) | Mixture of Experts | Sparse top-k routing: why only 2 of 8 experts activate per token |

---

## Setup

### 1. Install system dependencies

**macOS:**
```bash
brew install ffmpeg
brew install --cask mactex-no-gui   # LaTeX for math rendering
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y \
    libpango1.0-dev libcairo2-dev pkg-config python3-dev \
    ffmpeg texlive texlive-latex-extra
```

**Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html) and add it to PATH
- Install [MiKTeX](https://miktex.org/download) for LaTeX

### 2. Install Manim
```bash
pip install manim
```

### 3. Render an animation
```bash
# Low quality (fast preview) — 480p at 15fps
manim -ql lora_rank_decomposition.py LoRARankDecomposition

# High quality — 1080p at 60fps
manim -qh lora_rank_decomposition.py LoRARankDecomposition

# Open the video immediately after rendering
manim -ql --preview lora_rank_decomposition.py LoRARankDecomposition
```

Output videos are saved to `media/videos/<filename>/<quality>/`.

### 4. Render all animations
```bash
for f in *.py; do manim -ql "$f"; done
```

---

## Quality flags

| Flag | Resolution | FPS | Use for |
|---|---|---|---|
| `-ql` | 480p | 15 | Quick preview |
| `-qm` | 720p | 30 | Sharing drafts |
| `-qh` | 1080p | 60 | Final render |
| `-qk` | 2160p | 60 | 4K |

---

## Notes

- Render times range from ~30 seconds (low quality) to several minutes (high quality) per animation
- The `media/` output directory is gitignored — videos are not committed to the repo
- If LaTeX is not installed, `MathTex` scenes will fail; replace with `Text` as a fallback
