# CLI parameters and config mapping

This file is the source of truth for the command line interface. It maps each CLI flag to the corresponding `Config` field, shows the effective default (from `Config` when the flag is omitted), valid values, and the effect on behavior. At the end there is a compact list of settings that are only available via `Config` and not exposed in the CLI.

Order of precedence:
1) `--preset` applies first (fast, balanced, quality).
2) Any flags that follow override values from the preset.
3) Some flags derive other values (see `--workload`).

Notes:
- Paths can be relative to `src/` or absolute.
- You can pass multiple brushes by repeating `--brush` or by providing a comma separated list in a single flag.
- If a CLI flag is not provided, the system uses the default from `Config`.

## Presets

Current presets as implemented in the CLI:

| Name | workload_scale | make_video | max_size |
|---|---|---|---|
| fast | 4 | True | 640 |
| balanced | 25 | True | 0 |
| quality | 60 | True | 0 |

`--preset` only sets these fields. All other flags can still override them.

## Logger

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --log-level | (not in Config) | str | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL | Controls logging verbosity | Passed to `configure_logging` only |

## Paths

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --input | Config.input_image_path | str | ../assets/input/sky.jpg | file path | Source image to reconstruct | PNG or high quality JPG recommended |
| --out-image | Config.output_final_image_path | str | ../outputs/images/final.png | file path | Where final canvas is saved | Path is uniquified if it already exists |
| --out-video | Config.output_video_path | str | ../outputs/video/out.mp4 | file path | Where video is saved | Path is uniquified if it already exists |

## Brushes

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --brush | Config.brush_paths | list[str] | ../assets/brushes/brush.png | PNG files (grayscale) | Adds one or more brush masks | Repeat flag or use comma list |

## Core workload

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --workload | Config.workload_scale | int | 1 | > 0 | High level knob for compute budget | Also derives other fields (see below) |
| --make-video | Config.make_video | bool | True | flag | Enable video recording | Mutually exclusive with --no-video |
| --no-video | Config.make_video | bool | True | flag | Disable video recording | Overrides preset if used |

Derived by `--workload` (applied in CLI):
- `total_strokes = 1000 * workload_scale`
- `speed_fast_seconds = 2.64 * workload_scale`
- `speed_fast_to = 1.0 * workload_scale`

## Video cadence

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --video-fps | Config.video_fps | int | 30 | > 0 | Writer FPS | Used by video recorder |
| --save-every | Config.save_every_n_strokes | int or null | 10 | > 0 or null | Write a frame every N accepted strokes | Takes precedence over --video-duration |
| --video-duration | Config.video_duration_sec | int or null | 15 | >= 0 or null | Target duration used to choose frame stride | Used only if --save-every is not set |

## Input resize

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --max-size | Config.max_size | int | 0 | >= 0 | Downscale longest side to this many pixels | 0 keeps original size |

## Quality target

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --target-mse | Config.target_mse | float or null | null | > 0 or null | Early stop when global mean MSE <= value | Disabled if omitted |

## Size schedule

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --size-scale-mode | Config.size_scale_mode | str | log | log, linear | How to generate stroke box sizes across phases | |
| --levels | Config.levels | int | 5 | >= 2 | Number of size phases | |
| --largest-frac | Config.largest_frac | float | 0.35 | 0 < x <= 1 | Max box size as fraction of min(H, W) | |
| --smallest-px | Config.smallest_px | int | 10 | >= 1 | Minimum box size in pixels | Avoid 1 to reduce noise |

## ROI sampling

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --roi-sampling | Config.roi_sampling | str | topk_random | argmax, topk_random | How to pick ROI center | `topk` value is not exposed via CLI |

## Canvas and brush behavior

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --start-color-hex | Config.start_color_hex | str | #FFFFFF | hex | Initial canvas color | |
| --use-soft-edges / --no-soft-edges | Config.use_soft_edges | bool | True | flag | Soft edges use grayscale as coverage, hard edges use threshold | |
| --mask-threshold | Config.mask_threshold | float | 0.5 | 0 to 1 | Threshold for hard edges | Used only when soft edges are disabled |
| --use-alpha / --no-alpha | Config.use_alpha | bool | False | flag | Enable or disable global alpha multiplier | |
| --alpha-value | Config.alpha_value | float | 0.25 | 0 to 1 | Global alpha multiplier for weight mask | Used only when use_alpha is true |

## Determinism

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --seed | Config.seed | int | 42 | any int | Controls RNG for reproducibility | Affects ROI sampling, brush choice, angle when random |

## Preset selector

| Flag | Maps to | Type | Default (effective) | Values | Effect | Notes |
|---|---|---|---|---|---|---|
| --preset | (applies overrides) | str | none | fast, balanced, quality | Quick setup for workload and video | Later flags still override |

## Flags that map to multiple config fields

The `--workload` flag sets these fields in addition to `workload_scale`:
- `total_strokes`
- `speed_fast_seconds`
- `speed_fast_to`

## Config-only settings (no CLI flag)

These fields exist in `Config` but do not have a direct CLI flag. They can be changed by editing code or by constructing `Config` in Python.

| Name | Type | Default | Range | Effect | Used in |
|---|---|---|---|---|---|
| total_strokes | int | 1000 * workload_scale | > 0 | Hard cap on attempts | core.pipeline |
| orientation_mode | str | gradient | gradient, none, random | Stroke angle policy | strokes.stroke_engine |
| grad_min_strength | float | 1e-6 | >= 0 | Weak structure threshold for gradient-based angle | strokes.stroke_engine |
| angle_jitter_deg | float | 0.0 | >= 0 | Gaussian angle jitter | strokes.stroke_engine |
| use_cooldown | bool | True | bool | Enable cooldown priority map | strokes.roi_selection |
| cooldown_factor | float | 0.6 | 0 < x <= 1 | Downweight multiplier inside accepted ROI | strokes.roi_selection |
| cooldown_recover | float | 1.02 | > 0 | Per-step recovery multiplier | strokes.roi_selection |
| cooldown_min | float | 0.25 | 0 to 1 | Lower clip for cooldown weights | strokes.roi_selection |
| cooldown_max | float | 1.0 | 0 to 1 | Upper clip for cooldown weights | strokes.roi_selection |
| topk | int | 4096 | > 0 | K used by topk_random ROI sampling | strokes.roi_selection |
| record_last_hold_frames | int | 120 | >= 0 | Duplicate last frame at end of video | video.recorder, core.pipeline |
| postprocess_speed | bool | True | bool | Enable ffmpeg speed ramp postprocess | video.speed_ramp |
| speed_slow_seconds | float | 4.0 | >= 0 | Duration of slow start segment | video.speed_ramp |
| speed_fast_seconds | float | 2.64 * workload_scale | >= 0 | Duration of fast end segment | video.speed_ramp |
| speed_slow_from | float | 0.2 | > 0 | Start playback speed factor | video.speed_ramp |
| speed_fast_to | float | 1.0 * workload_scale | > 0 | End playback speed factor | video.speed_ramp |
| speed_steps | int | 10 | >= 1 | Smoothstep subsegments per edge | video.speed_ramp |
| speed_crf | int | 18 | >= 0 | x264 quality factor | video.speed_ramp |
| speed_preset | str | fast | x264 presets | Encoder speed vs compression | video.speed_ramp |
| speed_fps | float or null | null | >= 0 or null | Force FPS at encode time | video.speed_ramp |

## Usage examples

Fast preview (smaller input, short video):
```
painter --preset fast --input ./img/cat.jpg --out-video ./out/cat.mp4 --workload 3
```

High quality at full resolution with deterministic run:
```
painter --preset quality --input ./img/scene.png --seed 123 --no-video --target-mse 0.001
```

Multiple brushes:
```
painter --brush brushes/round.png --brush brushes/flat.png,brushes/rake.png
```
