from typing import Mapping, List
from enum import Enum

class MetricEnum(Enum):
    ACCURACY = 1
    F1 = 2
    RECALL = 3
    PRECISION = 4
    MATTHEW = 5
    BAS = 6
    LOSS = 7


class MetricsCalc:
    def __init__(this, name: str) -> None:
        this.reset()
        this.name = name

    def reset(this):
        this.sum: float = 0
        this.count: int = 0
        this.avg: float = 0
        this.value: float = 0

    def calculate_average(this) -> float:
        return this.sum / this.count

    def update(this, value: float, n: int = 1) -> None:
        this.value = value
        this.count += n
        this.sum += value * n
        this.avg = this.calculate_average()

class MetricsResult:
    def __init__(this) -> None:
        this.result: Mapping[str, MetricsCalc] = {}

    def get_update_avg(this, key: str) -> float:
        return this.result[key].avg

    def update(this, scores: Mapping[str, float], n: int):
        for score_name, value in scores.items():
            mCalc = MetricsCalc(name=score_name)
            mCalc.update(value, n)
            this.result[score_name] = mCalc



