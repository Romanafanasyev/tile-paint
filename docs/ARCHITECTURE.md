# Architecture

This page shows module boundaries, what is pure and what has side effects, and how data flows at runtime.

## Overview

- `cli` parses flags and builds `Config`.
- `core.pipeline` orchestrates the run: loads inputs, runs the engine, writes outputs, and optionally postprocesses video.
- `strokes.*` is the core engine and helpers. It is pure in the sense that it only manipulates in-memory arrays.
- `imagemath` provides small numeric helpers and is pure.
- `io.files` and `video.*` perform side effects: disk I O and ffmpeg subprocess.
- `config` defines validated parameters used everywhere.
- `logger` configures runtime logs.

## Module diagram

```mermaid
flowchart LR
  subgraph CLI
    C1[cli.__main__]
  end

  subgraph Core
    P1[core.pipeline]
  end

  subgraph Engine
    E1[strokes.stroke_engine]
    E2[strokes.roi_selection]
    E3[strokes.brush_ops]
    E4[strokes.geometry]
    E5[strokes.sizes]
  end

  subgraph Math
    M1[imagemath.image_math]
  end

  subgraph IO
    I1[io.files]
    V1[video.recorder]
    V2[video.speed_ramp]
  end

  CFG[config.config]
  LOG[logger.logger]

  C1 --> CFG
  C1 --> P1

  P1 --> I1
  P1 --> E1
  P1 --> V1
  P1 --> V2
  P1 --> LOG

  E1 --> E2
  E1 --> E3
  E1 --> E4
  E1 --> E5
  E1 --> M1
  E1 --> CFG

  E2 --> CFG
  E3 --> CFG
  E5 --> CFG

  I1 --> E3
  I1 --> P1

  classDef pure fill:#d6f5d6,stroke:#333,stroke-width:1px;
  classDef side fill:#ffe6cc,stroke:#333,stroke-width:1px;

  class E1,E2,E3,E4,E5,M1 pure;
  class I1,V1,V2 side;
  class P1 side;
  class C1 side;
```

Legend: green nodes are pure. Orange nodes perform side effects.

## Runtime data flow

1. CLI builds `Config` and calls the pipeline.
2. Pipeline loads the target image and brush masks from disk, converts them to float arrays.
3. Pipeline creates the cooldown map and the engine with the target, canvas, brushes, and RNG.
4. Pipeline runs the main loop:
   - recover cooldown
   - select ROI center using error map and cooldown
   - pick orientation
   - render and crop brush mask
   - build coverage and weight masks
   - pick color from target
   - blend and test MSE
   - on accept, commit and update error map, apply cooldown
   - maybe switch phase based on accept rate and attempt budget
5. Pipeline saves the final image and closes the video.
6. Optional: speed ramp postprocess replaces the video file in place.

## Dependency rules

- Engine modules depend on imagemath and config only. They do not touch disk or subprocess.
- IO and video modules do all side effects.
- Pipeline can call any module.
- Keep imagemath leaf pure.

This split keeps the core easy to unit test and the side effects easy to mock.
