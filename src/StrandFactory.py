import math
import os
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
    strand_count: int = 16
    strand_particle_count: int = 32
    strand_length: float = 2
    strand_length_variation: bool = False
    strand_length_variation_amount: float = 0.2

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
    strand_count: int
    pos: np.ndarray
    dir: np.ndarray
    uv0: np.ndarray


@dataclass
class Strands:
    strand_count: int
    strand_particle_count: int
    particle_positions: np.ndarray  # Flatted list of positions
    memory_layout: Utility.MemoryLayout


def generate_roots(settings: Settings) -> Roots:
    root_pos = np.zeros(settings.strand_count, dtype='object')
    root_dir = np.zeros(settings.strand_count, dtype='object')
    root_uv0 = np.zeros(settings.strand_count, dtype='object')

    # TODO: Other placement primitives

    if settings.primitive is PrimitiveType.Curtain:
        step = 1.0 / (settings.strand_count - 1.0)

        localDim = Utility.Vector3(1.0, 0.0, 0.0)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strand_count:
            uv = Utility.Vector2(i * step, 0.5)

            root_pos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, 0.0)
            root_dir[i] = localDir
            root_uv0[i] = uv

            i += 1

    if settings.primitive is PrimitiveType.StratifiedCurtain:
        step = 1.0 / settings.strand_count

        localDim = Utility.Vector3(1.0, 0.0, step)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strand_count:
            uvCell = Utility.Vector2(
                random.random(),
                random.random()
            )

            uv = Utility.Vector2((i + uvCell.x) * step, uvCell.y)

            root_pos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, localDim.z * (uv.y - 0.5))
            root_dir[i] = localDir
            root_uv0[i] = uv

            i += 1

    if settings.primitive is PrimitiveType.Brush:
        localDim = Utility.Vector3(1.0, 0.0, 1.0)
        localDir = Utility.Vector3(0.0, -1.0, 0.0)

        i = 0
        while i != settings.strand_count:
            uv = Utility.Vector2(random.random(), random.random())

            root_pos[i] = Utility.Vector3(localDim.x * (uv.x - 0.5), 0.0, localDim.z * (uv.y - 0.5))
            root_dir[i] = localDir
            root_uv0[i] = uv

            i += 1

    if settings.primitive is PrimitiveType.Cap:
        i = 0
        while i != settings.strand_count:
            localDir = Utility.random_sphere_direction()
            if localDir.y < 0:
                localDir.y = -localDir.y

            root_pos[i] = Utility.Vector3(localDir.x * 0.5, localDir.y * 0.5, localDir.z * 0.5)
            root_dir[i] = localDir
            root_uv0[i] = Utility.Vector2(localDir.x * 0.5 + 0.5, localDir.z * 0.5 + 0.5)

            i += 1

    return Roots(settings.strand_count, root_pos, root_dir, root_uv0)


def generate_strands(roots: Roots, settings: Settings) -> Strands:
    strand_pos = np.zeros(settings.strand_count * settings.strand_particle_count, dtype='object')

    particle_interval = settings.strand_length / (settings.strand_particle_count - 1)
    particle_interval_variation = settings.strand_length_variation_amount if settings.strand_length_variation else 0.0

    # Note: Don't need the list of this since we don't support alembic right now
    normalized_strand_length = 1.0
    normalized_strand_diameter = 1.0
    normalized_curl_radius = 1.0
    normalized_curl_slope = 1.0

    i = 0
    while i != settings.strand_count:
        step = normalized_strand_length * particle_interval  # * Utility.Lerp(1.0, random.random(), particle_interval_variation)

        cur_pos = roots.pos[i]
        cur_dir = roots.dir[i]

        begin, stride, end = Utility.get_strand_iterator(Utility.MemoryLayout.Interleaved,
                                                         i, settings.strand_count, settings.strand_particle_count)

        # TODO: Curls
        if settings.curl:
            cur_plane_u = Utility.Normalize(Utility.next_vector_in_plane(cur_dir))
            cur_plane_v = Utility.cross(cur_plane_u, cur_dir)

            target_radius = settings.curlRadius * 0.01
            target_slope = settings.curlSlope

            step_plane = step * math.cos(0.5 * math.pi * target_slope)

            if step_plane > 1.0 * target_radius:
                step_plane = 1.0 * target_radius

            if settings.curlSamplingStrategy == CurlSamplingStrategy.RelaxStrandLength:
                stepSlope = step * math.sin(0.5 * math.pi * target_slope)
            else:
                stepSlope = math.sqrt(step * step - step_plane * step_plane)

            a = 2.0 * math.asin(step_plane / (2.0 * target_radius)) if step_plane > 0.0 else 0.0
            t = 0

            cur_pos.x -= cur_plane_u.x * target_radius
            cur_pos.y -= cur_plane_u.y * target_radius
            cur_pos.z -= cur_plane_u.z * target_radius

            j = begin
            while j != end:
                du = target_radius * math.cos(t + a)
                dv = target_radius * math.sin(t * a)
                dn = stepSlope * t

                strand_pos[j] = Utility.Vector3(
                    cur_pos.x + (du * cur_plane_u.x) + (dv * cur_plane_v.x) + (dn * cur_dir.x),
                    cur_pos.y + (du * cur_plane_u.y) + (dv * cur_plane_v.y) + (dn * cur_dir.y),
                    cur_pos.z + (du * cur_plane_u.z) + (dv * cur_plane_v.z) + (dn * cur_dir.z)
                )

                j += stride
                t += 1
        else:
            k = 0
            j = begin
            while j != end:
                # TODO: Overload
                strand_pos[j] = Utility.Vector3(
                    cur_pos.x + (k * step * cur_dir.x),
                    cur_pos.y + (k * step * cur_dir.y),
                    cur_pos.z + (k * step * cur_dir.z)
                )

                # Looks like python array elements don't store a copy, not safe to modify it
                # cur_pos.x += step * curDir.x
                # cur_pos.y += step * curDir.y
                # cur_pos.z += step * curDir.z

                j += stride
                k += 1

        i += 1

    return Strands(settings.strand_count, settings.strand_particle_count, strand_pos, Utility.MemoryLayout.Interleaved)


# Build a strand group based on procedural settings
def build_procedural(settings: Settings = Settings()):
    # Generate the roots based on the primitive placement type.
    roots = generate_roots(settings)

    # Generate the strands based on the roots and other settings.
    return generate_strands(roots, settings)


# Build a strand group based on a line OBJ.
def build_from_asset(path):
    strand_pos = []
    strand_count = 0
    strand_particle_count = 0

    root = os.path.dirname(os.path.abspath(__file__))

    try:
        f = open("{}/data/{}.obj".format(root, path))
        for line in f:
            if line[0] == "v":
                i0 = line.find(" ") + 1
                i1 = line.find(" ", i0 + 1)
                i2 = line.find(" ", i1 + 1)

                # Technically pointless as the list is flattened before bound to GPU
                strand_pos.append(Utility.Vector3(
                    float(line[i0:i1]),
                    float(line[i1:i2]),
                    float(line[i2:-1])
                ))
            if line[0] == "l":
                connectivity = [int(s) for s in line.split() if s.isdigit()]
                strand_count += 1
                strand_particle_count = len(connectivity)
        f.close()
    except IOError:
        print("Failed to find file at path.")

    return Strands(strand_count, strand_particle_count, np.array(strand_pos), Utility.MemoryLayout.Sequential)
