import coalpy.gpu as gpu
import random
import math

from enum import Enum
from dataclasses import dataclass


class ClearMode(Enum):
    RAW    = 0
    UINT   = 1


s_clear_target      = gpu.Shader(file="utility/ClearTarget.hlsl",     name="ClearTarget",     main_function="ClearTarget")
s_clear_buffer_raw  = gpu.Shader(file="utility/ClearBufferRaw.hlsl",  name="ClearBufferRaw",  main_function="ClearBuffer")
s_clear_buffer_uint = gpu.Shader(file="utility/ClearBufferUInt.hlsl", name="ClearBufferUInt", main_function="ClearBuffer")

s_clear_buffer_shaders = {ClearMode.RAW:  s_clear_buffer_raw,
                          ClearMode.UINT: s_clear_buffer_uint}


class MemoryLayout:
    Sequential = 0
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


def random_sphere_direction():
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


def square_magnitude(v: Vector3) -> float:
    return v.x * v.x + v.y * v.y + v.z * v.z


def cross(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.z
    )


def dot(a: Vector3, b: Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def Normalize(v: Vector3) -> Vector3:
    m = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    return Vector3(
        v.x / m,
        v.y / m,
        v.z / m
    )


def project_on_plane(v: Vector3, n: Vector3) -> Vector3:
    d = dot(v, n)
    m = dot(n, n)
    return Vector3(
        v.x - n.x * d / m,
        v.y - n.y * d / m,
        v.z - n.z * d / m
    )


def next_vector_in_plane(n) -> Vector3:
    while True:
        r = project_on_plane(random_sphere_direction(), n)
        if square_magnitude(r) > 1e-5:
            break
    return r


def lerp(a: float, b: float, t: float) -> float:
    return (t * a) + ((1 - t) * b)


def get_strand_iterator(memoryLayout, strandIndex, strandCount, strandParticleCount):
    if memoryLayout is MemoryLayout.Sequential:
        strandParticleBegin = strandIndex * strandParticleCount
        strandParticleStride = 1
    else:
        strandParticleBegin = strandIndex
        strandParticleStride = strandCount

    strandParticleEnd = strandParticleBegin + strandParticleStride * strandParticleCount

    return strandParticleBegin, strandParticleStride, strandParticleEnd


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def divup(a, b):
    return int((a + b - 1)/b)


def alignup(a, b):
    return divup(a, b) * b


def clear_target(cmd, color, target, w, h):
    cmd.dispatch(
        shader=s_clear_target,
        constants=color,
        x=math.ceil(w / 8),
        y=math.ceil(h / 8),
        z=1,
        outputs=target
    )


def clear_buffer(cmd, value, count, target, mode):
    cmd.dispatch(
        shader=s_clear_buffer_shaders[mode],

        constants=[
            int(value),
            int(count)
        ],

        outputs=target,

        x=math.ceil(count / 64),
        y=1,
        z=1
    )