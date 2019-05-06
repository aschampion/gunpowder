import logging
import numpy as np
import string
import random

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
    '''Squeeze a singleton dimension out of batches. Can be matched with an
    :ref:`Expand` node downstream, which can be constructed via the `expand`
    helper method.

    Args:

        axes (``dict`, :class:`ArrayKey` -> ``int``)

            An mapping of axes indices to squeeze for each for specified array.
    '''

    def __init__(
            self,
            axes):

        self.axes = {array_key: {'axis': ax, 'propagate': True} for (array_key, ax) in axes.items()}

    def __squeeze_spec(self, spec, axis):

        sq_spec = ArraySpec()
        if spec.roi is not None:
            (offset, roi_offset_el) = squeeze_coordinate(spec.roi.get_offset(), axis)
            (shape, roi_shape_el) = squeeze_coordinate(spec.roi.get_shape(), axis)
            spec.roi = Roi(offset, shape)
            sq_spec.roi = Roi((roi_offset_el,), (roi_shape_el,))
        if spec.voxel_size is not None:
            (spec.voxel_size, voxel_size_el) = squeeze_coordinate(spec.voxel_size, axis)
            sq_spec.voxel_size = Coordinate((voxel_size_el,))

        return sq_spec

    def expand(self):

        return Expand(squeeze_node=self)

    def setup(self):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            spec = self.spec[array_key].copy()
            ax_props['spec'] = self.__squeeze_spec(spec, axis)

            self.updates(array_key, spec)

            ax_props['req_key'] = ArrayKey('_' + str(array_key) + '_squeeze_' +
                ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(32)]))
            self.provides(ax_props['req_key'], ax_props['spec'])

    def prepare(self, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            req = request[array_key]
            if ax_props['req_key'] in request:
                squeeze_req = request[ax_props['req_key']]
            else:
                squeeze_req = ax_props['spec']
            offset = expand_coordinate(req.roi.get_offset(), axis, squeeze_req.roi.get_offset()[0])
            shape = expand_coordinate(req.roi.get_shape(), axis, squeeze_req.roi.get_shape()[0])
            req.roi = Roi(offset, shape)
            if req.voxel_size is not None:
                req.voxel_size = expand_coordinate(req.voxel_size, axis, squeeze_req.voxel_size[0])

    def process(self, batch, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            array = batch.arrays[array_key]
            assert array.data.shape[axis] == 1, "Attempt to expand/squeeze non-singleton dimension (%s size %s) of array (%s)"\
                % (axis, array.data.shape[axis], array_key)
            array.data = array.data.squeeze(axis=axis)
            self.__squeeze_spec(array.spec, axis)

            if ax_props['req_key'] in request:
                squeeze_req = request[ax_props['req_key']]
            else:
                squeeze_req = ax_props['spec']

            data = np.array([0])
            batch[ax_props['req_key']] = Array(data, spec=squeeze_req)


class Expand(BatchFilter):
    '''Expand a singleton dimension into batches. May be matched with an
    :ref:`Squeeze` node upstream which has removed the singleton dimension
    which will be reinserted.

    Args:

        squeeze_node (:class:``Squeeze``)

            The node that removed the singleton dimension which will be
            reinsterted.
    '''

    def __init__(
            self,
            axes={},
            squeeze_node=None):

        if squeeze_node:
            self.axes = {**axes, **squeeze_node.axes}
        else:
            self.axes = {**axes}

        for (array_key, ax_props) in self.axes.items():
            if not ax_props['propagate']:
                ax_props['req_key'] = ArrayKey('_' + str(array_key) + '_squeeze' +
                    ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(32)]))

    def stubs(self):

        return StubArrayProvider({p['req_key']: p['spec'] for (_, p) in self.axes.items() if not p['propagate']})

    def setup(self):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            spec = self.spec[array_key].copy()
            self.__expand_spec(spec, axis, ax_props['spec'])

            self.updates(array_key, spec)

    def __expand_spec(self, spec, axis, sq_spec):

        if spec.roi is not None:
            offset = expand_coordinate(spec.roi.get_offset(), axis, sq_spec.roi.get_offset()[0])
            shape = expand_coordinate(spec.roi.get_shape(), axis, sq_spec.roi.get_shape()[0])
            spec.roi = Roi(offset, shape)
        if spec.voxel_size is not None:
            spec.voxel_size = expand_coordinate(spec.voxel_size, axis, sq_spec.voxel_size[0])

    def prepare(self, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            req = request[array_key]
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
                spec.voxel_size = ax_props['spec'].voxel_size

            request[ax_props['req_key']] = spec

    def process(self, batch, request):

        for (array_key, ax_props) in self.axes.items():
            axis = ax_props['axis']
            array = batch.arrays[array_key]
            array.data = np.expand_dims(array.data, axis=axis)
            sq_spec = batch[ax_props['req_key']].spec
            self.__expand_spec(array.spec, axis, sq_spec)


class StubArrayProvider(BatchFilter):
    def __init__(self, arrays):

        self.arrays = arrays

    def setup(self):

        for (key, spec) in self.arrays.items():
            self.provides(key, spec)

    def process(self, batch, request):

        for (key, _) in self.arrays.items():
            data = np.array([0])
            batch[key] = Array(data, spec=request[key])
