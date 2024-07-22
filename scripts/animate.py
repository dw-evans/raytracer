from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes import (
        Csys,
        Mesh,
        Sphere,
        Material,
        Camera,
    )


from typing import TypeVar, Callable
import copy
from math import sin, cos, tan, pi

T = TypeVar("T")
FuncOfFloat = Callable[[T, float], T]


class Animation:
    """
    Class that stores animation information about a scene...
    It should be able to...
        Store the initial state of the animation

        Animate a coordinate system
            Position
            Orientation
                Point towards another csys
        Animate a camera
        Animate a mesh
            Position
            Scale?
        Animate material
            Colour
            Transparency


    What if the animation..


    Attributes:
        function: callable[float]
            stores the function, of time, that creates the state as a function of time.

    Methods:

    next():
        Calculates the next position/orientation/state. Perhaps off the initial state plus x*dt
        to mitigate rounding errors.
        Updates the object to its next state

    reset():
        resets the object to its initial state

    evaluate();


    """

    def __init__(self, obj: T, fun: FuncOfFloat) -> None:
        self.obj = obj
        self.fun = fun
        # self.initial_state = copy.deepcopy(self.obj)

    def evaluate(self, t: float) -> T:
        return self.fun(self.obj, t)

    def reset(self):
        """Resets the state of the object to as initialized"""
        raise NotImplementedError
        obj = self.obj
        for key, val in self.initial_state.__dict__.items():
            setattr(obj, key, val)
