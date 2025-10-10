# Brushes

This is a short guide for preparing and validating brush masks.

## What a brush is

A brush is a grayscale mask where white means paint and black means skip. During rendering the mask is scaled so that its longest side matches the current stroke box size and then rotated by the chosen angle. New pixels created by rotation are filled with black.

## File format and size

- Use PNG. Single channel is enough, we load images as mode L.
- Keep source resolution between 256 and 1024 px on the longest side. Larger inputs rotate and downscale better.
- Keep the background black rather than transparent. Alpha is ignored.

## Soft vs hard edges

- Soft edges: if `use_soft_edges` is true, the grayscale level acts as subpixel coverage. This produces natural blending.
- Hard edges: if `use_soft_edges` is false, the mask is thresholded at `mask_threshold`. Use high-contrast shapes.
- Global alpha: if `use_alpha` is true, `alpha_value` multiplies the weight mask. This is a simple way to make all strokes more translucent.

## Design tips

- Provide variety: round, flat, rake, and textured shapes.
- Avoid tiny single-pixel specks that become noise at small sizes.
- Keep edges antialiased for soft mode. For hard mode keep edges crisp.
- Use descriptive filenames. Group families in subfolders if you plan to ship multiple brushes.

## How to pass multiple brushes

- Repeat the `--brush` CLI flag: `--brush a.png --brush b.png`.
- Or pass a comma separated list once: `--brush a.png,b.png,c.png`.

## QA checklist

- Open the mask and check that white really means paint and black means skip.
- Dry run with a small image, for example `--max-size 320 --workload 2`, and observe how the brush behaves.
- Try `--use-soft-edges` and `--no-soft-edges` to verify both paths.
- Check rotation artifacts at 0, 45, and 90 degrees.
