# Explaining the implementation

The notebook loop remains recognizable in `CareMonitor.run()`: OpenCV reads frames,
YOLO finds the selected person, cvzone estimates joints, and rules propose a state.
The application then builds intervals, inspects context, detects events and saves JSON.
Classes hold changing state such as model instances, selected ID and movement history.
Dataclasses carry observations and intervals. There is no training pipeline to explain.

## Existing networks

YOLO26s is a pretrained single-stage detector producing object boxes and scores.
ByteTrack associates person detections over time. Bed detection is separate so a
class-agnostic tracker cannot transfer a furniture track to a person.

cvzone wraps MediaPipe Pose, which estimates 33 landmarks. This application applies
it to the selected person's padded crop, then maps coordinates back to the image.
The pinned package uses synchronous inference, preventing previous-frame results
from being paired with a current frame. One coordinate scale preserves joint angles.

Qwen3-VL optionally receives three chronological panels in one image through Ollama.
A visual encoder represents the image for a language model, which proposes an
activity and confidence in a two-field JSON response.
The proposal does not replace event rules. All weights are pretrained. Review the
primary model documentation before discussing deeper architectural details:

- https://docs.ultralytics.com/models/yolo26/
- https://github.com/cvzone/cvzone/blob/master/cvzone/PoseModule.py
- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

## Design choices

Walking and out-of-bed overlap, so each interval has one activity and one bed status.
UNKNOWN time remains explicit rather than being assigned to out-of-bed duration.
The sum of activity durations equals the analyzed media duration.

A single frame beside the bed cannot establish an exit. The offline agent examines
neighboring intervals and requires walking-away evidence. Following context would
introduce delay in a live version. Every review and abstention is logged in the report.

The code is deliberately conservative about target identity and unknown periods.
This reduces unsupported events but causes misses in short or occluded sequences.
The measured evaluation exposes those tradeoffs rather than hiding them.

## With more time

Collect independently annotated clips with caregivers, blankets and prolonged absence.
Use calibrated mattress segmentation. Train a small activity classifier over pose
sequences on development subjects only. Add appearance-based re-identification.
Measure VLM latency and accuracy impact separately; fewer UNKNOWN labels alone do
not prove improvement.
