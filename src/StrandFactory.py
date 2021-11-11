import numpy as np
import random

from src import Utility
from dataclasses import dataclass

class CurlSamplingStrategy:
    RelaxStrandLength = 0
    RelaxCurlSlope = 1

class PrimitiveType:
    Curtain = 0
    Brush = 1
    Cap = 2

@dataclass
class Settings:
    # General
    primitive: PrimitiveType = PrimitiveType.Cap
    strandCount: int = 64
    strandParticleCount: int = 32
    strandLength: float = 0.25
    strandLengthVariation: bool = False
    strandLengthVariationAmount: float = 0.2

    # Curls
    curl: bool = False
    curlRadius: float = 1.0
    curlSlope: float = 0.3
    curlVariation: bool = False
    curlVariationRadius: float = 0.1
    curlVariationSlope: float = 0.3
    curlSamplingStrategy: CurlSamplingStrategy = CurlSamplingStrategy.RelaxStrandLength

@dataclass
class Roots:
    strandCount:   int
    pos: np.ndarray
    dir: np.ndarray
    uv0: np.ndarray

@dataclass
class Strands:
    strandCount:         int
    strandParticleCount: int
    particlePositions:   np.ndarray
    memoryLayout:        Utility.MemoryLayout

def GenerateRoots(settings : Settings) -> Roots:

    rootPos = np.empty(settings.strandCount, dtype='object')
    rootDir = np.empty(settings.strandCount, dtype='object')
    rootUV0 = np.empty(settings.strandCount, dtype='object')

    # TODO: Other placement primitives

    if settings.primitive is PrimitiveType.Curtain:
        step = 1.0 / (settings.strandCount - 1.0)

        localDim = Utility.Vector3(1.0,  0.0, 0.0)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strandCount:
            uv = Utility.Vector2(i * step, 0.5)

            rootPos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, 0.0)
            rootDir[i] = localDir
            rootUV0[i] = uv

            i += 1

    return Roots(settings.strandCount, rootPos, rootDir, rootUV0)

def GenerateStrands(roots : Roots, settings : Settings) -> Strands:

    strandPos = np.empty(settings.strandCount * settings.strandParticleCount, dtype='object')

    particleInterval = settings.strandLength / (settings.strandParticleCount - 1)
    particleIntervalVariation = settings.strandLengthVariationAmount if settings.strandLengthVariation else 0.0

    # Note: Don't need the list of this since we don't support alembic right now
    normalizedStrandLength   = 1.0
    normalizedStrandDiameter = 1.0
    normalizedCurlRadius     = 1.0
    normalizedCurlSlope      = 1.0

    i = 0
    while i != settings.strandCount:
        step = normalizedStrandLength * particleInterval * Utility.Lerp(1.0, random.random(), particleIntervalVariation)

        curPos = roots.pos[i]
        curDir = roots.dir[i]

        begin, stride, end = Utility.GetStrandIterator(Utility.MemoryLayout.Interleaved,
                                                       i, settings.strandCount, settings.strandParticleCount)

        # TODO: Curls

        j = begin
        while j != end:
            strandPos[j] = curPos

            # TODO: Overloads
            curPos.x += step * curDir.x
            curPos.y += step * curDir.y
            curPos.z += step * curDir.z

            j += stride

        i += 1

    return Strands(settings.strandCount, settings.strandParticleCount, strandPos, Utility.MemoryLayout.Sequential)

def Build(settings: Settings):

    # Temp: Manually set to curtain
    settings.primitive = PrimitiveType.Curtain

    # Generate the roots based on the primitive placement type.
    roots = GenerateRoots(settings)

    # Generate the strands based on the roots and other settings.
    return GenerateStrands(roots, settings)