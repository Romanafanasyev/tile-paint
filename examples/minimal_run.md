# Picking workload and running examples

Goal: run good results in one command. The main knob is `--workload` (thousands of strokes). Everything else is secondary.

- `--workload N` means `total_strokes = 1000 * N`.
- Runtime scales roughly linearly with `workload` and with image pixels (megapixels). If you double workload, expect about 2x time. If you double resolution, expect about 2x time.
- Presets are just shortcuts. You can ignore them and set `--workload` directly. Use a preset only if you like its defaults (for example `fast` downsizes to 640px on the long side).

How to run (from `src/`):
```
python -m painter <flags...>
```

## What images work best vs worst

- Easiest: smooth gradients and soft content (sky, fog, water, large color areas). Little fine detail.
- Medium: natural photos with moderate detail (portraits, animals, landscapes without tiny structures).
- Hard: lots of razor-sharp edges and tiny high-contrast features (black/white graphics, thin line art, dense hatching). Even huge workloads will not be perfect.

## Rule of thumb: choose workload by content and resolution

Pick the row for your image size (long side), then the column for content type. Start at the lower bound, bump up if you still see blocky edges or missing texture.

| Long side | Smooth gradients | Natural photo (moderate detail) | Heavy detail or sharp edges |
|---|---|---|---|
| 512–768 px | 1–3 | 15–25 | 40–60 |
| 1080–1440 px | 2–5 | 25–40 | 60–90 |
| 2000–3000 px | 5–8 | 40–70 | 90–140 |

Tips when pushing detail:
- Increase `--levels` (more phases) and decrease `--smallest-px` (smaller strokes).
- For crisp black/white content use `--no-soft-edges` and tune `--mask-threshold` (0.5–0.7).
- If progress stalls, consider `--target-mse` to stop early.

## Some sample images in `assets/input`

- `grad.jpeg`, `sky.jpg`: low detail. Workload 1 is enough at default size.
- `cat.jpg`, `lion.jpg`: moderate detail. Use workload 25 or higher.
- `lines.jpg`: worst case by design (many hard edges). Even workload 60 is barely okay. Use hard edges and smaller strokes.

### Commands you can paste

Smooth gradient demos (fast and clean):
```
python -m painter --input ../assets/input/grad.jpeg --workload 1 --out-image ../outputs/images/grad_w1.png --out-video ../outputs/video/grad_w1.mp4
python -m painter --input ../assets/input/sky.jpg  --workload 1 --out-image ../outputs/images/sky_w1.png  --out-video ../outputs/video/sky_w1.mp4
```

Natural photos (needs more strokes):
```
python -m painter --input ../assets/input/cat.jpg  --workload 25 --out-image ../outputs/images/cat_w25.png  --out-video ../outputs/video/cat_w25.mp4
python -m painter --input ../assets/input/lion.jpg --workload 25 --out-image ../outputs/images/lion_w25.png --out-video ../outputs/video/lion_w25.mp4
```

Hard black/white lines (tuned for crisper edges):
```
python -m painter --input ../assets/input/lines.jpg --workload 60 --no-soft-edges --mask-threshold 0.6 --smallest-px 4 --levels 7 --out-image ../outputs/images/lines_w60_hard.png --out-video ../outputs/video/lines_w60_hard.mp4
```

## Presets: when to use

- `--preset fast`: downsizes to 640px for quick previews. Combine with a small workload (1–3) for instant iterations.
- `--preset balanced`: full resolution, higher default workload. Good baseline for photos.
- `--preset quality`: full resolution, high workload baseline. Use when you want the best result and can wait.

You can always override workload. Examples:
```
# Fast preview with downsize
python -m painter --preset fast --input ../assets/input/cat.jpg --workload 5

# Full-res quality
python -m painter --preset quality --input ../assets/input/lion.jpg --workload 30
```

## Time planning

- Time ~ workload × megapixels. If a run with workload 20 on a 1 MP image takes T, then:
  - workload 40 on the same image takes ~2T
  - workload 20 on a 2 MP image takes ~2T
- Video adds minor overhead. Disable with `--no-video` if you only need the final image.

## Repro and stopping

- Add `--seed 123` to reproduce a run exactly (same workload, same inputs, same flags).
- If you only need a certain quality, set `--target-mse` to stop early once the global error is low enough.
