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

def SquareMagnitude(v : Vector3) -> float:
    return v.x * v.x + v.y * v.y + v.z * v.z

def Cross(a : Vector3, b : Vector3) -> Vector3:
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.z
    )

def Dot(a : Vector3, b : Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def Normalize(v : Vector3) -> Vector3:
    m = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    return Vector3(
        v.x / m,
        v.y / m,
        v.z / m
    )

def ProjectOnPlane(v: Vector3, n: Vector3) -> Vector3:
    d = Dot(v, n)
    m = Dot(n, n)
    return Vector3(
        v.x - n.x * d / m,
        v.y - n.y * d / m,
        v.z - n.z * d / m
    )

def NextVectorInPlane(n) -> Vector3:
    while True:
        r = ProjectOnPlane(RandomSphereDirection(), n)
        if SquareMagnitude(r) > 1e-5:
            break
    return r

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