"""Layout and moving obstacles for the Service Hall."""

from dataclasses import dataclass
from math import ceil, dist


Point = tuple[float, float]


@dataclass(frozen=True)
class Arena:
    size: tuple[int, int]
    start: Point
    goal: Point
    bollards: tuple[Point, ...]
    movers: tuple[tuple[Point, Point, float], ...]  # center, travel vector, radians/second
    max_steps: int
    walls: tuple[tuple[Point, Point], ...]
    preferred_clearance: float

    def static_centers(self):
        centers = list(self.bollards)
        # Model each partition with overlapping unit circles, including at the ends.
        for first, last in self.walls:
            intervals = ceil(dist(first, last))
            for index in range(intervals + 1):
                t = index / intervals
                centers.append(tuple(a + t * (b - a) for a, b in zip(first, last)))
        return centers


SERVICE_HALL = Arena(
    size=(48, 26),
    start=(4.5, 5.0),
    goal=(43.5, 21.5),
    bollards=((22.5, 11.0), (36.5, 5.5), (39.5, 18.5)),
    movers=(
        ((20.0, 17.0), (0.0, 3.7), 0.58),
        ((35.0, 13.0), (2.7, 0.0), 0.71),
    ),
    max_steps=480,
    walls=(((13.0, 0.0), (13.0, 13.0)), ((29.5, 12.5), (29.5, 26.0))),
    preferred_clearance=3.5,
)
