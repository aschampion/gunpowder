import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import Array, ArrayKey
from gunpowder.coordinate import Coordinate
from gunpowder.array_spec import ArraySpec
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)


def squeeze_coordinate(coord, axis):

    coord = np.array(coord)
    el = coord[axis]
    return (Coordinate(np.delete(coord, axis)), el)

def expand_coordinate(coord, axis, el):

    coord = np.array(coord)
    return Coordinate(np.insert(coord, axis, el))


class Squeeze(BatchFilter):
    '''Squeeze a singleton dimension out of batches. Must be matched with an
    :ref:`Expand` node downstream, which can be constructed via the `expand`
    helper method.

    Args:

        axes (``dict`, :class:`ArrayKey` -> ``int``)

            An mapping of axes indices to squeeze for each for specified array.
    '''

    def __init__(
            self,
            axes):

        self.axes = {array_key: {'axis': ax} for (array_key, ax) in axes.items()}

    def squeeze_spec(self, spec, axis):

        if spec.roi is not None:
            (offset, roi_offset_el) = squeeze_coordinate(spec.roi.get_offset(), axis)
            (shape, roi_shape_el) = squeeze_coordinate(spec.roi.get_shape(), axis)
            spec.roi = Roi(offset, shape)
        if spec.voxel_size is not None:
            (spec.voxel_size, voxel_size_el) = squeeze_coordinate(spec.voxel_size, axis)

        return (roi_offset_el, roi_shape_el, voxel_size_el)

    def expand(self):

        return Expand(self)

    def setup(self):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            spec = self.spec[array_key].copy()
            (ax_props['roi_offset'],
             ax_props['roi_shape'],
             ax_props['voxel_shape']) = self.squeeze_spec(spec, axis)

            self.updates(array_key, spec)

            ax_props['req_key'] = ArrayKey('_' + str(array_key) + '_squeeze')
            spec = ArraySpec()

            spec.roi = Roi((ax_props['roi_offset'],), (ax_props['roi_shape'],))
            spec.voxel_size = Coordinate((ax_props['voxel_shape'],))
            self.provides(ax_props['req_key'], spec)

    def prepare(self, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            req = request[array_key]
            squeeze_req = request[ax_props['req_key']]
            offset = expand_coordinate(req.roi.get_offset(), axis, squeeze_req.roi.get_offset()[0])
            shape = expand_coordinate(req.roi.get_shape(), axis, squeeze_req.roi.get_shape()[0])
            req.roi = Roi(offset, shape)
            print('req', req.roi, req.voxel_size)
            if req.voxel_size is not None:
                req.voxel_size = expand_coordinate(req.voxel_size, axis, squeeze_req.voxel_size[0])

    def process(self, batch, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            array = batch.arrays[array_key]
            print(array.data.shape)
            assert array.data.shape[axis] == 1, "Attempt to expand/squeeze non-singleton dimension (%s size %s) of array (%s)"\
                % (axis, array.data.shape[axis], array_key)
            array.data = array.data.squeeze(axis=axis)
            self.squeeze_spec(array.spec, axis)
            print('foobar', array.spec, array.data.shape)

            data = np.array([0])
            batch[ax_props['req_key']] = Array(data, spec=request[ax_props['req_key']])


class Expand(BatchFilter):
    '''Squeeze a singleton dimension out of batches. Must be matched with an
    :ref:`Expand` node downstream, which can be constructed via the `expand`
    helper method.

    Args:

        axes (``dict`, :class:`ArrayKey` -> ``int``)

            An mapping of axes indices to squeeze for each for specified array.
    '''

    def __init__(
            self,
            squeeze_node):

        self.squeeze_node = squeeze_node

    def setup(self):

        for (array_key, ax_props) in self.squeeze_node.axes.items():
            axis = ax_props['axis']
            spec = self.spec[array_key].copy()
            print('before', spec)
            self.expand_spec(spec, axis, ax_props['roi_offset'], ax_props['roi_shape'], ax_props['voxel_shape'])

            self.updates(array_key, spec)
            print('after', spec)

    def expand_spec(self, spec, axis, offset_el, shape_el, voxel_size_el):

        if spec.roi is not None:
            offset = expand_coordinate(spec.roi.get_offset(), axis, offset_el)
            shape = expand_coordinate(spec.roi.get_shape(), axis, shape_el)
            spec.roi = Roi(offset, shape)
        if spec.voxel_size is not None:
            spec.voxel_size = expand_coordinate(spec.voxel_size, axis, voxel_size_el)

    def prepare(self, request):

        for (array_key, ax_props) in self.squeeze_node.axes.items():
            axis = ax_props['axis']
            req = request[array_key]
            print('exp prep req', req.roi, req.voxel_size)
            voxel_shape_el = None
            (offset, roi_offset_el) = squeeze_coordinate(req.roi.get_offset(), axis)
            (shape, roi_shape_el) = squeeze_coordinate(req.roi.get_shape(), axis)
            req.roi = Roi(offset, shape)
            if req.voxel_size is not None:
                (req.voxel_size, voxel_shape_el) = squeeze_coordinate(req.voxel_size, axis)

            spec = ArraySpec()
            spec.roi = Roi((roi_offset_el,), (roi_shape_el,))
            if voxel_shape_el is not None:
                spec.voxel_size = Coordinate((voxel_shape_el))
            else:
                spec.voxel_size = Coordinate((ax_props['voxel_shape'],))
            print('exp prep req sq', spec.roi, spec.voxel_size)

            request[ax_props['req_key']] = spec

    def process(self, batch, request):

        for (array_key, ax_props) in self.squeeze_node.axes.items():
            axis = ax_props['axis']
            array = batch.arrays[array_key]
            array.data = np.expand_dims(array.data, axis=axis)
            sq = batch[ax_props['req_key']]
            self.expand_spec(array.spec, axis,
                sq.spec.roi.get_offset()[0], sq.spec.roi.get_shape()[0], sq.spec.voxel_size[0])
            print('bazbar', array.spec, array.data.shape)
