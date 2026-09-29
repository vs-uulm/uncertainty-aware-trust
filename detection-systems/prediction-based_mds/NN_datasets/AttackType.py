from enum import Enum


class AttackType(Enum):
    accelerationMultiplication = 0
    constantPositionOffset = 1
    constantSpeedOffset = 2
    dataReplay = 3
    dosAttack = 4
    feignedBraking = 5
    positionMirroring = 6
    randomPositionOffset = 7
    randomSpeedOffset = 8
    reversedHeading = 9
    suddenConstantSpeed = 10
    suddenStop = 11
    timeDelayAttack = 12
    trafficCongestionSybil = 13
    zeroSpeedReport = 14