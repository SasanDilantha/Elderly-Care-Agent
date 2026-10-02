import argparse
import json
from pathlib import Path

from elderly_care_agent.models import Settings


def main():
    parser = argparse.ArgumentParser(description="YOLO + cvzone elderly activity monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze a video and save a JSON report")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--show", action="store_true")
    analyze.add_argument("--output", type=Path, default=Path("outputs/demo"))
    evaluate = commands.add_parser("evaluate", help="Measure results on annotated clips")
    evaluate.add_argument("--config", type=Path, default=Path("config/evaluation.yml"))
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    for command in [analyze, evaluate]:
        command.add_argument("--sample-fps", type=float, default=5)
        command.add_argument("--model", type=Path)
        command.add_argument("--with-vlm", action="store_true")
        command.add_argument("--vlm-model", default="qwen3-vl:4b-instruct")
        command.add_argument("--max-reviews", type=int, default=3)
    prepare = commands.add_parser("prepare-dataset", help="Download and stage the curated dataset")
    prepare.add_argument("--source", type=Path, help="Existing extracted GMDCSA24 directory")
    args = parser.parse_args()
    try:
        if args.command == "prepare-dataset":
            from elderly_care_agent.dataset import Dataset

            print(json.dumps(Dataset().prepare(args.source), indent=2))
            return
        from elderly_care_agent.pipeline import CareMonitor

        settings = Settings(
            sample_fps=args.sample_fps, vlm_model=args.vlm_model, max_reviews=args.max_reviews
        )
        if args.sample_fps <= 0 or args.max_reviews < 0:
            raise ValueError("Sample FPS must be positive and review limit nonnegative")
        monitor = CareMonitor(settings, args.model)
        if args.command == "evaluate":
            from elderly_care_agent.evaluation import Evaluator

            result = Evaluator().run(args.config, monitor, args.output, args.with_vlm)
        else:
            result = monitor.run(args.video, show=args.show, use_vlm=args.with_vlm)
            result = monitor.save(result, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
