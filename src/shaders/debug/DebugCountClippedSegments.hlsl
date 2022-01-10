ByteAddressBuffer _SegmentCountBuffer : register(t0);
RWBuffer<uint>    _FrustumCulledCount : register(u0);

[numthreads(64, 1, 1)]
void CountClippedSegments(uint3 dispatchThreadID : SV_DispatchThreadID,
                          uint groupIndex : SV_GroupIndex)
{
    uint count = _SegmentCountBuffer.Load(4 * dispatchThreadID.x);
    GroupMemoryBarrierWithGroupSync();

    uint waveCount = WaveActiveSum(count);

    if (groupIndex == 0)
        InterlockedAdd(_FrustumCulledCount[0], waveCount);
}