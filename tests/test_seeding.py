from retention_lab.utils.seeding import numpy_rng, stream_seed, torch_generator


def test_stream_seeds_are_stable():
    assert stream_seed(1234, "model-init") == stream_seed(1234, "model-init")


def test_stream_seeds_differ_across_names_and_seeds():
    assert stream_seed(1234, "model-init") != stream_seed(1234, "packing-order")
    assert stream_seed(1234, "model-init") != stream_seed(1235, "model-init")


def test_numpy_streams_reproduce():
    a = numpy_rng(7, "x").integers(0, 1000, size=16)
    b = numpy_rng(7, "x").integers(0, 1000, size=16)
    assert (a == b).all()


def test_torch_generators_reproduce():
    import torch

    a = torch.rand(8, generator=torch_generator(7, "x"))
    b = torch.rand(8, generator=torch_generator(7, "x"))
    assert torch.equal(a, b)
