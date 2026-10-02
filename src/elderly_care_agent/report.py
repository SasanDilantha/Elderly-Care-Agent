import json
from pathlib import Path


class Report:
    def __init__(self, result):
        self.summary = {
            key: result[key]
            for key in (
                "observation_duration_sec",
                "activity_duration_sec",
                "bed_exit_count",
                "bed_return_count",
                "total_in_bed_sec",
                "total_out_of_bed_sec",
                "longest_out_of_bed_period_sec",
                "final_state",
            )
        }
        self.events = [
            dict(
                event=row["event"],
                start_time=self.timestamp(row["start_sec"]),
                confirm_time=self.timestamp(row["confirmed_sec"]),
                previous_state=row["previous_state"],
                current_state=row["current_state"],
                confidence=row["confidence"],
                decision=row["decision"],
            )
            for row in result["events"]
        ]
        self.additional = {key: value for key, value in result.items() if key not in self.summary}

    @staticmethod
    def timestamp(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("report.json", self.summary),
            ("events.json", self.events),
            ("additional.json", self.additional),
        ):
            (directory / name).write_text(
                json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
            )
        lines = [
            f"{row['start']:07.2f} - {row['end']:07.2f}  {row['state'].upper()}"
            for row in self.additional["timeline"]
        ]
        (directory / "timeline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dict(
            summary=self.summary,
            events=self.events,
            additional_outputs=dict(
                details=str(directory / "additional.json"),
                timeline=str(directory / "timeline.txt"),
            ),
        )
