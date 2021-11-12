import math
import coalpy.gpu as gpu
import random
import math

from dataclasses import dataclass

ShaderClearTarget = gpu.Shader(file = "ClearTarget.hlsl", name = "ClearTarget", main_function = "Main")


class MemoryLayout:
    Sequential  = 0
    Interleaved = 1

@dataclass
class Vector2:
    x: float
    y: float

@dataclass
class Vector3:
    x: float
    y: float
    z: float

#    def __rmul__(self, scalar):
#        x = self.x + scalar
#        y = self.y + scalar
#        z = self.z + scalar
#        return Vector3(x, y, z)
#
#    def __iadd__(self, other):
#        x = self.x + other.x
#        y = self.y + other.y
#        z = self.z + other.z
#        return Vector3(x, y, z)

def RandomSphereDirection():
    rnd = Vector2(
        random.random(),
        random.random()
    )
    z = -1 + 2 * rnd.x
    r = math.sqrt(max(1.0 - z * z, 0.0))
    angle = rnd.y * math.pi * 2
    s = math.sin(angle)
    c = math.cos(angle)
    return Vector3(c * r, s * r, z)

def Lerp(a : float, b : float, t : float) -> float:
    return (t * a) + ((1 - t) * b)

def GetStrandIterator(memoryLayout, strandIndex, strandCount, strandParticleCount):
    if memoryLayout is MemoryLayout.Sequential:
        strandParticleBegin  = strandIndex * strandParticleCount
        strandParticleStride = 1
    else:
        strandParticleBegin  = strandIndex
        strandParticleStride = strandCount

    strandParticleEnd = strandParticleBegin + strandParticleStride * strandParticleCount

    return (strandParticleBegin, strandParticleStride, strandParticleEnd)


def ClearTarget(cmd, color, target, w, h):
    cmd.dispatch(
        shader = ShaderClearTarget,
        constants = color,
        x = math.ceil(w / 8),
        y = math.ceil(h / 8),
        z = 1,
        outputs = target
    )