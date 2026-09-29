import unittest

from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    ObservationSource,
)
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import (
    GroundTruthAnnotation,
    StateSegment,
    TimeRange,
)


def create_segment(
    start_sec: float,
    end_sec: float,
    activity: ActivityState,
    occupancy: BedOccupancy,
) -> StateSegment:
    return StateSegment(
        time_range=TimeRange(start_sec, end_sec),
        activity=activity,
        bed_occupancy=occupancy,
        confidence=1.0,
        source=ObservationSource.GROUND_TRUTH,
    )


class TimeRangeTests(unittest.TestCase):
    def test_duration_and_overlap(self) -> None:
        first = TimeRange(0.0, 5.0)
        touching = TimeRange(5.0, 8.0)

        self.assertEqual(5.0, first.duration_sec)
        self.assertFalse(first.overlaps(touching))

    def test_empty_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "greater than"):
            TimeRange(2.0, 2.0)


class StateSegmentTests(unittest.TestCase):
    def test_impossible_activity_occupancy_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "cannot be out of bed"):
            create_segment(
                0.0,
                1.0,
                ActivityState.LYING_IN_BED,
                BedOccupancy.OUT_OF_BED,
            )


class GroundTruthAnnotationTests(unittest.TestCase):
    def test_timeline_aggregates_activity_and_occupancy_durations(self) -> None:
        annotation = GroundTruthAnnotation(
            video_id="demo",
            duration_sec=10.0,
            segments=(
                create_segment(
                    0.0,
                    4.0,
                    ActivityState.LYING_IN_BED,
                    BedOccupancy.IN_BED,
                ),
                create_segment(
                    4.0,
                    10.0,
                    ActivityState.WALKING,
                    BedOccupancy.OUT_OF_BED,
                ),
            ),
        )

        self.assertEqual(4.0, annotation.activity_durations()[ActivityState.LYING_IN_BED])
        self.assertEqual(6.0, annotation.activity_durations()[ActivityState.WALKING])
        self.assertEqual(4.0, annotation.occupancy_durations()[BedOccupancy.IN_BED])

    def test_timeline_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "contiguous"):
            GroundTruthAnnotation(
                video_id="demo",
                duration_sec=10.0,
                segments=(
                    create_segment(
                        0.0,
                        4.0,
                        ActivityState.LYING_IN_BED,
                        BedOccupancy.IN_BED,
                    ),
                    create_segment(
                        5.0,
                        10.0,
                        ActivityState.WALKING,
                        BedOccupancy.OUT_OF_BED,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()

