import numpy as np
import unittest
import faiss


class TestMoveToGpu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = faiss.StandardGpuResources()

    def create_index(self, factory_string):
        dimension = 128
        n = 2500
        db_vectors = np.random.random((n, dimension)).astype('float32')
        index = faiss.index_factory(dimension, factory_string)
        index.train(db_vectors)
        index.add(db_vectors)
        return index

    def verify_throws_on_unsupported_index(self, factory_string):
        idx = self.create_index(factory_string)
        try:
            faiss.index_cpu_to_gpu(self.res, 0, idx)
        except Exception as e:
            if "not implemented" not in str(e):
                self.fail("Expected an exception but no exception was "
                          "thrown for factory_string: %s." % factory_string)

    def verify_succeeds_on_supported_index(self, factory_string):
        idx = self.create_index(factory_string)
        try:
            faiss.index_cpu_to_gpu(self.res, 0, idx)
        except Exception as e:
            self.fail("Unexpected exception thrown factory_string: "
                      "%s; error message: %s." % (factory_string, str(e)))

    def test_index_cpu_to_gpu_unsupported_indices(self):
        self.verify_throws_on_unsupported_index("PQ16")
        self.verify_throws_on_unsupported_index("LSHrt")
        self.verify_throws_on_unsupported_index("HNSW,PQ16")

    def test_index_cpu_to_gpu_supported_indices(self):
        self.verify_succeeds_on_supported_index("Flat")
        self.verify_succeeds_on_supported_index("IVF1,Flat")
        self.verify_succeeds_on_supported_index("IVF32,SQ8")
        self.verify_succeeds_on_supported_index("IVF32,PQ8")
