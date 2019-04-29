from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class SqueezeTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 30, 100)),
                voxel_size=(20, 2, 2)))
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((20100,2010,2010), (1800,10,80)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        batch = Batch()

        # have the pixels encode their position
        for (array_key, spec) in request.array_specs.items():

            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)

        return batch

class TestSqueeze(ProviderTest):

    def test_output(self):

        source = SqueezeTestSource()

        chunk_request = BatchRequest()
        chunk_request.add(ArrayKeys.RAW, (400,2,34))
        chunk_request.add(ArrayKeys.GT_LABELS, (200,2,14))

        squeeze = Squeeze({ArrayKeys.RAW: 1, ArrayKeys.GT_LABELS: 1})
        expand = squeeze.expand()

        pipeline = source + squeeze + expand.stubs() + expand + Scan(chunk_request, num_workers=10)

        with build(pipeline):

            raw_spec = pipeline.spec[ArrayKeys.RAW]
            labels_spec = pipeline.spec[ArrayKeys.GT_LABELS]

            full_request = BatchRequest({
                    ArrayKeys.RAW: raw_spec,
                    ArrayKeys.GT_LABELS: labels_spec
                }
            )

            batch = pipeline.request_batch(full_request)
            voxel_size = pipeline.spec[ArrayKeys.RAW].voxel_size

        comb_valid_roi = None
        for (array_key, array) in batch.arrays.items():
            roi = array.spec.roi
            min_valid = roi.get_begin() - chunk_request[array_key].roi.get_begin()
            max_valid = roi.get_shape() - chunk_request[array_key].roi.get_shape() + (1,)*roi.dims()
            valid_roi = Roi(min_valid, max_valid)
            if comb_valid_roi is None:
                comb_valid_roi = valid_roi
            else:
                comb_valid_roi = comb_valid_roi.intersect(valid_roi)
        valid_sl = list(map(slice, comb_valid_roi.get_begin()//voxel_size, comb_valid_roi.get_end()//voxel_size))

        # assert that pixels encode their position
        for (array_key, array) in batch.arrays.items():

            if array_key not in (ArrayKeys.RAW, ArrayKeys.GT_LABELS):
                continue

            # the z,y,x coordinates of the ROI
            roi = array.spec.roi
            meshgrids = np.meshgrid(
                    range(roi.get_begin()[0]//voxel_size[0], roi.get_end()[0]//voxel_size[0]),
                    range(roi.get_begin()[1]//voxel_size[1], roi.get_end()[1]//voxel_size[1]),
                    range(roi.get_begin()[2]//voxel_size[2], roi.get_end()[2]//voxel_size[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            self.assertTrue((array.data[valid_sl] == data[valid_sl]).all())

        assert(batch.arrays[ArrayKeys.RAW].spec.roi.get_offset() == (20000, 2000, 2000))

        # test scanning with empty request

        squeeze = Squeeze({ArrayKeys.RAW: 1, ArrayKeys.GT_LABELS: 1})

        pipeline = source + squeeze + squeeze.expand() + Scan(chunk_request, num_workers=10)
        with build(pipeline):
            batch = pipeline.request_batch(BatchRequest())
