import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from .batch_filter import BatchFilter
from gunpowder.array import Array

logger = logging.getLogger(__name__)

class MapLabels(BatchFilter):
    '''Map label values.

    The labels will be replaced by background_value. An optional ignore mask
    will be created and set to 0 for the excluded locations that are further
    than a threshold away from not excluded locations.

    Args:

        labels (:class:`ArrayKey`):

            The array containing the labels.

        label_map (``dict``, ``int`` -> ``Number``):
    '''

    def __init__(
            self,
            labels,
            label_map,
            sparse=False,
            output_dtype=None):

        self.labels = labels
        self.label_map = label_map
        self.sparse = sparse
        self.output_dtype = output_dtype

    def setup(self):

        if self.output_dtype:
            spec = self.spec[self.labels].copy()
            spec.dtype = self.output_dtype
            self.updates(self.labels, spec)

        if not self.sparse:
            label_in = list(self.label_map.keys())
            label_out = list(self.label_map.values())
            self.label_map_dense = np.zeros(len(label_in), dtype=self.spec[self.labels].dtype)
            self.label_map_dense[label_in] = label_out

    def process(self, batch, request):

        gt = batch.arrays[self.labels]

        if self.sparse:
            out = gt.data.astype(self.dtype)

            for l in self.label_map:
                out[gt.data == l] = self.label_map[l]
        else:
            if self.output_dtype is not None and self.output_dtype != gt.data.dtype:
                out = np.empty(gt.data.shape, dtype=self.output_dtype)
            else:
                out = gt.data

            np.take(self.label_map_dense, gt.data, out=out)

        gt.data[:] = out[:]
