import math
import coalpy.gpu as gpu

from dataclasses import dataclass

ShaderClearTarget = gpu.Shader(file = "shaders/ClearTarget.hlsl", name = "ClearTarget", main_function = "Main")


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

def Lerp(a : float, b : float, t : float) -> float:
    return 0

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