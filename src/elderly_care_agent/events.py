from elderly_care_agent.models import State


class BedEvents:
    def __init__(self, settings):
        self.settings = settings

    def detect(self, segments, observations):
        events = []
        in_bed = None
        departure = None
        arrival = None
        previous = State.UNKNOWN
        person_id = None
        for segment in segments:
            observed = [row for row in observations if segment.start <= row.time < segment.end]
            identity = next((row.person_id for row in observed if row.person_id is not None), None)
            if person_id is not None and identity is not None and identity != person_id:
                in_bed, departure, arrival = None, None, None
            if segment.state == State.UNKNOWN or segment.source == "vlm":
                visible = (
                    observed
                    and person_id is not None
                    and all(row.person_id == person_id for row in observed)
                )
                if segment.duration > self.settings.event_gap + 1e-6 or not visible:
                    in_bed, departure, arrival = None, None, None
                continue
            person_id = identity
            state = segment.state
            if state.bed == "in_bed":
                departure = None
                if in_bed is False and arrival is None:
                    arrival = (segment.start, previous)
                if (
                    arrival
                    and state == State.LYING
                    and segment.duration >= self.settings.event_hold
                ):
                    events.append(
                        self.event(
                            "bed_return",
                            arrival[0],
                            segment.start + self.settings.event_hold,
                            arrival[1],
                            state,
                            segment.confidence,
                            "NORMAL",
                        )
                    )
                    arrival = None
                in_bed = True
            else:
                arrival = None
                if in_bed is True and departure is None:
                    departure = (segment.start, previous)
                if departure and state == State.WALKING:
                    nearby = [
                        row
                        for row in observations
                        if departure[0] <= row.time < segment.end
                        and (row.features or {}).get("foot_distance") is not None
                        and row.state != State.UNKNOWN
                    ]
                    confirmed = self.confirm_departure(nearby, segment.start)
                    if confirmed is not None:
                        events.append(
                            self.event(
                                "bed_exit",
                                departure[0],
                                confirmed,
                                departure[1],
                                state,
                                segment.confidence,
                                "MONITOR",
                            )
                        )
                        departure = None
                in_bed = False
            previous = state
        return events

    def confirm_departure(self, observations, segment_start):
        if not observations:
            return None
        origin = observations[0].features
        margin = 0.25 * origin["torso_length"]
        walking_since = None
        away_samples = 0
        for row in observations:
            walking_since = (
                (row.time if walking_since is None else walking_since)
                if row.state == State.WALKING
                else None
            )
            distance = row.features["foot_distance"]
            away = (
                row.state == State.WALKING
                and distance < -margin
                and distance < origin["foot_distance"] - margin
            )
            away_samples = away_samples + 1 if away else 0
            if (
                away_samples >= 2
                and walking_since is not None
                and row.time - max(segment_start, walking_since) >= self.settings.event_hold - 1e-6
            ):
                return row.time
        return None

    @staticmethod
    def event(kind, start, confirmed, previous, current, confidence, decision):
        return dict(
            event=kind,
            start_sec=start,
            confirmed_sec=confirmed,
            previous_state=previous,
            current_state=current,
            confidence=confidence,
            decision=decision,
        )


class AlertPolicy:
    def __init__(self, settings):
        self.settings = settings

    def decide(self, segments, events):
        decisions = []
        outside_start = None
        sitting_start = None
        for row in segments:
            sitting_start = (
                (row.start if sitting_start is None else sitting_start)
                if row.state == State.BED_SITTING
                else None
            )
            if row.state.bed == "out_of_bed" and row.source != "vlm":
                outside_start = row.start if outside_start is None else outside_start
                exits = [
                    event
                    for event in events
                    if event["event"] == "bed_exit"
                    and outside_start <= event["confirmed_sec"] <= row.end
                ]
                if exits and row.end - outside_start >= self.settings.absence_alert:
                    decisions.append(
                        dict(
                            decision="ALERT",
                            time=max(
                                outside_start + self.settings.absence_alert,
                                exits[0]["confirmed_sec"],
                            ),
                            reason="Confirmed bed exit followed by prolonged visible absence",
                        )
                    )
            else:
                outside_start = None
            if row.state == State.UNKNOWN and row.duration >= self.settings.unknown_monitor:
                decisions.append(
                    dict(
                        decision="MONITOR",
                        time=row.start + self.settings.unknown_monitor,
                        reason="Activity remains uncertain",
                    )
                )
            if (
                sitting_start is not None
                and row.end - sitting_start >= self.settings.sitting_monitor
            ):
                decisions.append(
                    dict(
                        decision="MONITOR",
                        time=sitting_start + self.settings.sitting_monitor,
                        reason="Prolonged sitting on bed; edge position needs review",
                    )
                )
        decisions.extend(
            dict(decision="MONITOR", time=e["confirmed_sec"], reason="Confirmed bed exit")
            for e in events
            if e["event"] == "bed_exit"
        )
        rank = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}
        return max(
            decisions,
            key=lambda item: rank[item["decision"]],
            default=dict(decision="NORMAL", time=0.0, reason="No configured condition observed"),
        )
