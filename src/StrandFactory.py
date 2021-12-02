import math

import numpy as np
import random

from src import Utility
from dataclasses import dataclass

class CurlSamplingStrategy:
    RelaxStrandLength = 0
    RelaxCurlSlope = 1

class PrimitiveType:
    Curtain = 0
    StratifiedCurtain = 1
    Brush = 2
    Cap = 3

@dataclass
class Settings:
    # General
    primitive: PrimitiveType = PrimitiveType.Brush
    strandCount: int = 16
    strandParticleCount: int = 32
    strandLength: float = 2
    strandLengthVariation: bool = False
    strandLengthVariationAmount: float = 0.2

    # Curls
    curl: bool = True
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
    particlePositions:   np.ndarray # Flatted list of positions
    memoryLayout:        Utility.MemoryLayout

def GenerateRoots(settings : Settings) -> Roots:

    rootPos = np.zeros(settings.strandCount, dtype='object')
    rootDir = np.zeros(settings.strandCount, dtype='object')
    rootUV0 = np.zeros(settings.strandCount, dtype='object')

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

    if settings.primitive is PrimitiveType.StratifiedCurtain:
        step = 1.0 / settings.strandCount

        localDim = Utility.Vector3(1.0, 0.0, step)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strandCount:
            uvCell = Utility.Vector2(
                random.random(),
                random.random()
            )

            uv = Utility.Vector2((i + uvCell.x) * step, uvCell.y)

            rootPos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, localDim.z * (uv.y - 0.5))
            rootDir[i] = localDir
            rootUV0[i] = uv

            i += 1

    if settings.primitive is PrimitiveType.Brush:
        localDim = Utility.Vector3(1.0, 0.0, 1.0)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strandCount:
            uv = Utility.Vector2(random.random(), random.random())

            rootPos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, localDim.z * (uv.y - 0.5))
            rootDir[i] = localDir
            rootUV0[i] = uv

            i += 1

    if settings.primitive is PrimitiveType.Cap:
        i = 0
        while i != settings.strandCount:
            localDir = Utility.RandomSphereDirection()
            if localDir.y < 0:
                localDir.y = -localDir.y

            rootPos[i] = Utility.Vector3(localDir.x * 0.5, localDir.y * 0.5, localDir.z * 0.5)
            rootDir[i] = localDir
            rootUV0[i] = Utility.Vector2(localDir.x * 0.5 + 0.5, localDir.z * 0.5 + 0.5)

            i += 1


    return Roots(settings.strandCount, rootPos, rootDir, rootUV0)

def GenerateStrands(roots : Roots, settings : Settings) -> Strands:

    strandPos = np.zeros(settings.strandCount * settings.strandParticleCount, dtype='object')

    particleInterval = settings.strandLength / (settings.strandParticleCount - 1)
    particleIntervalVariation = settings.strandLengthVariationAmount if settings.strandLengthVariation else 0.0

    # Note: Don't need the list of this since we don't support alembic right now
    normalizedStrandLength   = 1.0
    normalizedStrandDiameter = 1.0
    normalizedCurlRadius     = 1.0
    normalizedCurlSlope      = 1.0

    i = 0
    while i != settings.strandCount:
        step = normalizedStrandLength * particleInterval # * Utility.Lerp(1.0, random.random(), particleIntervalVariation)

        curPos = roots.pos[i]
        curDir = roots.dir[i]

        begin, stride, end = Utility.GetStrandIterator(Utility.MemoryLayout.Interleaved,
                                                       i, settings.strandCount, settings.strandParticleCount)

        # TODO: Curls
        if settings.curl:
            curPlaneU = Utility.Normalize(Utility.NextVectorInPlane(curDir))
            curPlaneV = Utility.Cross(curPlaneU, curDir)

            targetRadius = settings.curlRadius * 0.01
            targetSlope = settings.curlSlope

            stepPlane = step * math.cos(0.5 * math.pi * targetSlope)

            if stepPlane > 1.0 * targetRadius:
                stepPlane = 1.0 * targetRadius

            if settings.curlSamplingStrategy == CurlSamplingStrategy.RelaxStrandLength:
                stepSlope = step * math.sin(0.5 * math.pi * targetSlope)
            else:
                stepSlope = math.sqrt(step * step - stepPlane * stepPlane)

            a = 2.0 * math.asin(stepPlane / (2.0 * targetRadius)) if stepPlane > 0.0 else 0.0
            t = 0

            curPos.x -= curPlaneU.x * targetRadius
            curPos.y -= curPlaneU.y * targetRadius
            curPos.z -= curPlaneU.z * targetRadius

            j = begin
            while j != end:
                du = targetRadius * math.cos(t + a)
                dv = targetRadius * math.sin(t * a)
                dn = stepSlope * t

                strandPos[j] = Utility.Vector3(
                    curPos.x + (du * curPlaneU.x) + (dv * curPlaneV.x) + (dn * curDir.x),
                    curPos.y + (du * curPlaneU.y) + (dv * curPlaneV.y) + (dn * curDir.y),
                    curPos.z + (du * curPlaneU.z) + (dv * curPlaneV.z) + (dn * curDir.z)
                )

                j += stride
                t += 1
        else:
            k = 0
            j = begin
            while j != end:

                # TODO: Overload
                strandPos[j] = Utility.Vector3(
                    curPos.x + (k * step * curDir.x),
                    curPos.y + (k * step * curDir.y),
                    curPos.z + (k * step * curDir.z)
                )

                # Looks like python array elements don't store a copy, not safe to modify it
                # curPos.x += step * curDir.x
                # curPos.y += step * curDir.y
                # curPos.z += step * curDir.z

                j += stride
                k += 1

        i += 1

    return Strands(settings.strandCount, settings.strandParticleCount, strandPos, Utility.MemoryLayout.Interleaved)

def Build(settings: Settings = Settings()):
    # Generate the roots based on the primitive placement type.
    roots = GenerateRoots(settings)

    # Generate the strands based on the roots and other settings.
    return GenerateStrands(roots, settings)