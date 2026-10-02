from elderly_care_agent.models import Segment, State


class Timeline:
    def __init__(self, settings):
        self.settings = settings

    def build(self, observations, duration):
        segments = []
        for index, row in enumerate(observations):
            end = observations[index + 1].time if index + 1 < len(observations) else duration
            if segments and segments[-1].state == row.state:
                segments[-1].end = end
                segments[-1].confidence = min(segments[-1].confidence, row.confidence)
            else:
                segments.append(Segment(row.time, end, row.state, row.confidence))
        for segment in segments:
            if segment.duration < self.settings.confirmation:
                segment.state, segment.confidence = State.UNKNOWN, 0.0
        return self.merge(segments)

    @staticmethod
    def merge(segments):
        merged = []
        for row in segments:
            if merged and merged[-1].state == row.state and merged[-1].source == row.source:
                merged[-1].end = row.end
                merged[-1].confidence = min(merged[-1].confidence, row.confidence)
            else:
                merged.append(Segment(row.start, row.end, row.state, row.confidence, row.source))
        return merged
