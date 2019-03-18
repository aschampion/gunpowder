import logging
import numpy as np
import os
import skimage.io

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class DirectorySource(BatchProvider):
    '''
    Args:

        filename (``string``):

            The input file.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''
    def __init__(
            self,
            path,
            datasets,
            array_specs):

        self.path = path
        self.datasets = datasets
        self.array_specs = array_specs
        self.data = {}

    def setup(self):
        for (array_key, ds_name) in self.datasets.items():

            filename = os.path.join(self.path, ds_name)

            if not os.path.isfile(filename):
                raise RuntimeError("%s not in %s" % (ds_name, self.path))

            spec = self.__read_image(array_key, filename)

            self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for (array_key, request_spec) in request.array_specs.items():
            logger.debug("Reading %s in %s...", array_key, request_spec.roi)

            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = Array(
                self.__read(array_key, dataset_roi),
                array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_image(self, array_key, filename):

        dataset = skimage.io.imread(filename)
        print(dataset.shape)
        print('foooufjhalkj')

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = Coordinate((1,)*len(dataset.shape))
            logger.warning("WARNING: No resolution information "
                           "for %s (dataset %s), voxel size has been set to %s. This "
                           "might not be what you want.",
                           array_key, ds_name, spec.voxel_size)
            spec.voxel_size = voxel_size

        ndims = len(spec.voxel_size)

        if spec.roi is None:
            offset = Coordinate((0,)*ndims)

            shape = Coordinate(dataset.shape[-ndims:])
            spec.roi = Roi(offset, shape*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s" %
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, filename, spec.dtype,
                           spec.interpolatable)

        self.data[array_key] = dataset
        print(spec)

        return spec

    def __read(self, array_key, roi):
        c = len(self.data[array_key].shape) - len(self.spec[array_key].voxel_size)
        return self.data[array_key][(slice(None),)*c + roi.to_slices()]
