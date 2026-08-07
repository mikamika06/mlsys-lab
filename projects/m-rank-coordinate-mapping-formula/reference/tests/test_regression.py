import sys
sys.path.insert(0, ".")
from mesh.mapping import rank_to_coords, coords_to_rank
from mesh.subgroup import get_subgroup_ranks
from mesh.reconstruct import reconstruct_mesh_shape


def test_mapping_roundtrip():
    shape = (2, 4, 8)
    for r in range(64):
        c = rank_to_coords(r, shape)
        assert coords_to_rank(c, shape) == r


def test_subgroup_membership():
    shape = (2, 4)
    ranks = get_subgroup_ranks(shape, {0: 1})
    assert ranks == [4, 5, 6, 7]


def test_reconstruct_shape():
    ranks = list(range(64))
    assert reconstruct_mesh_shape(ranks) == (2, 4, 8)
