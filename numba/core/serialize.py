"""
Serialization support for compiled functions.
"""
import sys
import abc
import io
import copyreg


import pickle
from numba import cloudpickle


#
# Pickle support
#

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


# Keep unpickled object via `numba_unpickle` alive.
_unpickled_memo = {}


def _numba_unpickle(address, bytedata, hashed):
    """Used by `numba_unpickle` from _helperlib.c
    Parameters
    ----------
    address : int
    bytedata : bytes
    hashed : bytes
    Returns
    -------
    obj : object
        unpickled object
    """
    key = (address, hashed)
    try:
        obj = _unpickled_memo[key]
    except KeyError:
        _unpickled_memo[key] = obj = cloudpickle.loads(bytedata)
    return obj


def dumps(obj):
    """Similar to `pickle.dumps()`. Returns the serialized object in bytes.
    """
    pickler = NumbaPickler
    with io.BytesIO() as buf:
        p = pickler(buf, protocol=4)
        p.dump(obj)
        pickled = buf.getvalue()

    return pickled


# Alias to pickle.loads to allow `serialize.loads()`
loads = cloudpickle.loads


class _CustomPickled:
    """A wrapper for objects that must be pickled with `NumbaPickler`.
    Standard `pickle` will pick up the implementation registered via `copyreg`.
    This will spawn a `NumbaPickler` instance to serialize the data.
    `NumbaPickler` overrides the handling of this type so as not to spawn a
    new pickler for the object when it is already being pickled by a
    `NumbaPickler`.
    """

    __slots__ = 'ctor', 'states'

    def __init__(self, ctor, states):
        self.ctor = ctor
        self.states = states

    def _reduce(self):
        return _CustomPickled._rebuild, (self.ctor, self.states)

    @classmethod
    def _rebuild(cls, ctor, states):
        return cls(ctor, states)


def _unpickle__CustomPickled(serialized):
    """standard unpickling for `_CustomPickled`.
    Uses `NumbaPickler` to load.
    """
    ctor, states = loads(serialized)
    return _CustomPickled(ctor, states)


def _pickle__CustomPickled(cp):
    """standard pickling for `_CustomPickled`.
    Uses `NumbaPickler` to dump.
    """
    serialized = dumps((cp.ctor, cp.states))
    return _unpickle__CustomPickled, (serialized,)


# Register custom pickling for the standard pickler.
copyreg.pickle(_CustomPickled, _pickle__CustomPickled)


def custom_reduce(cls, states):
    """For customizing object serialization in `__reduce__`.
    Object states provided here are used as keyword arguments to the
    `._rebuild()` class method.
    Parameters
    ----------
    states : dict
        Dictionary of object states to be serialized.
    Returns
    -------
    result : tuple
        This tuple conforms to the return type requirement for `__reduce__`.
    """
    return custom_rebuild, (_CustomPickled(cls, states),)


def custom_rebuild(custom_pickled):
    """Customized object deserialization.
    This function is referenced internally by `custom_reduce()`.
    """
    cls, states = custom_pickled.ctor, custom_pickled.states
    return cls._rebuild(**states)


def is_serialiable(obj):
    """Check if *obj* can be serialized.
    Parameters
    ----------
    obj : object
    Returns
    --------
    can_serialize : bool
    """
    with io.BytesIO() as fout:
        pickler = NumbaPickler(fout)
        try:
            pickler.dump(obj)
        except pickle.PicklingError:
            return False
        else:
            return True


def _no_pickle(obj):
    raise pickle.PicklingError(f"Pickling of {type(obj)} is unsupported")


def disable_pickling(typ):
    """This is called on a type to disable pickling
class SlowNumbaPickler(pickle._Pickler):
    """Extends the pure-python Pickler to support the pickling need in Numba.
    Adds pickling for closure functions, modules.
    Adds customized pickling for _CustomPickled to avoid invoking a new
    Pickler instance.
    Note: this is used on Python < 3.8 unless `pickle5` is installed.
    Note: This is good for debugging because the C-pickler hides the traceback
    """
    NumbaPickler.disabled_types.add(typ)
    # The following is needed for Py3.7
    NumbaPickler.dispatch_table[typ] = _no_pickle
    # Return `typ` to allow use as a decorator
    return typ


class NumbaPickler(cloudpickle.CloudPickler):
    disabled_types = set()
    """A set of types that pickling cannot is disabled.
    """

    def reducer_override(self, obj):
        # Overridden to disable pickling of certain types
        if type(obj) in self.disabled_types:
            _no_pickle(obj)  # noreturn
        return super().reducer_override(obj)


def _custom_reduce__custompickled(cp):
    return cp._reduce()


NumbaPickler.dispatch_table[_CustomPickled] = _custom_reduce__custompickled


class ReduceMixin(abc.ABC):
    """A mixin class for objects that should be reduced by the NumbaPickler instead
    of the standard pickler.
    """
    # Subclass MUST override the below methods

    @abc.abstractmethod
    def _reduce_states(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def _rebuild(cls, **kwargs):
        raise NotImplementedError

    # Subclass can override the below methods

    def _reduce_class(self):
        return self.__class__

    # Private methods

    def __reduce__(self):
        return custom_reduce(self._reduce_class(), self._reduce_states())

class PickleCallableByPath:
    """Wrap a callable object to be pickled by path to workaround limitation
    in pickling due to non-pickleable objects in function non-locals.

    Note:
    - Do not use this as a decorator.
    - Wrapped object must be a global that exist in its parent module and it
      can be imported by `from the_module import the_object`.
# ----------------------------------------------------------------------------
# The following code is adapted from cloudpickle as of
# https://github.com/cloudpipe/cloudpickle/commit/9518ae3cc71b7a6c14478a6881c0db41d73812b8    # noqa: E501
# Please see LICENSE.third-party file for full copyright information.

def _is_importable(obj):
    """Check if an object is importable.
    Parameters
    ----------
    obj :
        Must define `__module__` and `__qualname__`.
    """
    if obj.__module__ in sys.modules:
        ptr = sys.modules[obj.__module__]
        # Walk through the attributes
        parts = obj.__qualname__.split('.')
        if len(parts) > 1:
            # can't deal with function insides classes yet
            return False
        for p in parts:
            try:
                ptr = getattr(ptr, p)
            except AttributeError:
                return False
        return obj is ptr
    return False


def _function_setstate(obj, states):
    """The setstate function is executed after creating the function instance
    to add `cells` into it.
    """
    cells = states.pop('cells')
    for i, v in enumerate(cells):
        _cell_set(obj.__closure__[i], v)
    return obj


def _reduce_function_no_cells(func, globs):
    """_reduce_function() but return empty cells instead.
    """
    if func.__closure__:
        oldcells = [cell.cell_contents for cell in func.__closure__]
        cells = [None for _ in range(len(oldcells))] # idea from cloudpickle
    else:
        oldcells = ()
        cells = None
    rebuild_args = (_reduce_code(func.__code__), globs, func.__name__, cells,
                    func.__defaults__)
    return rebuild_args, oldcells


def _cell_rebuild(contents):
    """Rebuild a cell from cell contents
    """
    if contents is None:
        return CellType()
    else:
        return CellType(contents)

    Usage:

    >>> def my_fn(x):
    >>>     ...
    >>> wrapped_fn = PickleCallableByPath(my_fn)
    >>> # refer to `wrapped_fn` instead of `my_fn`
    """
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __reduce__(self):
        return type(self)._rebuild, (self._fn.__module__, self._fn.__name__,)

    @classmethod
    def _rebuild(cls, modname, fn_path):
        return cls(getattr(sys.modules[modname], fn_path))
=======
def _cell_set(cell, value):
    """Set *value* into *cell* because `.cell_contents` is not writable
    before python 3.7.
    See https://github.com/cloudpipe/cloudpickle/blob/9518ae3cc71b7a6c14478a6881c0db41d73812b8/cloudpickle/cloudpickle.py#L298   # noqa: E501
    """
    if PYVERSION >= (3, 7):  # pragma: no branch
        cell.cell_contents = value
    else:
        _cell_set = FunctionType(
            _cell_set_template_code, {}, '_cell_set', (), (cell,),)
        _cell_set(value)


def _make_cell_set_template_code():
    """See _cell_set"""
    def _cell_set_factory(value):
        lambda: cell
        cell = value

    co = _cell_set_factory.__code__

    _cell_set_template_code = CodeType(
        co.co_argcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_cellvars,  # co_freevars is initialized with co_cellvars
        (),  # co_cellvars is made empty
    )
    return _cell_set_template_code


if PYVERSION < (3, 7):
    _cell_set_template_code = _make_cell_set_template_code()

# End adapting from cloudpickle
# ----------------------------------------------------------------------------
