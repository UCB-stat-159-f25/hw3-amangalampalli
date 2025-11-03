import numpy as np
import ligotools.readligo as rl

def test_dq_channel_to_seglist_basic():
    channel = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    fs = 4
    segs = rl.dq_channel_to_seglist(channel, fs=fs)

    # Expect two slices
    assert len(segs) == 2

    assert isinstance(segs[0], slice)
    assert segs[0].start == 1 * fs
    assert segs[0].stop == 3 * fs  # end-exclusive

    assert isinstance(segs[1], slice)
    assert segs[1].start == 4 * fs
    assert segs[1].stop == 5 * fs

def test_dq2segs_with_default_channel_dict():
    dq_mask = np.array([0, 1, 1, 0], dtype=int)  # true on seconds 1001 and 1002
    gps_start = 1000
    seglist = rl.dq2segs({"DEFAULT": dq_mask}, gps_start=gps_start)

    # Should be a SegmentList with one (start, stop) tuple
    assert isinstance(seglist, rl.SegmentList)
    assert len(seglist.seglist) == 1
    assert seglist.seglist[0] == (1001, 1003)  # end is exclusive
