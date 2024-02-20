from enum import Enum


class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"  # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"  # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration


################################################################################


class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Explicit dynamics model
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = (  # PyBullet physics update with ground effect, drag, and downwash
        "pyb_gnd_drag_dw"
    )


################################################################################


class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0  # Red, green, blue (and alpha)
    DEP = 1  # Depth
    SEG = 2  # Segmentation by object id
    BW = 3  # Black and white


################################################################################


class ActionType(Enum):
    """Action type enumeration class."""

    RPM = "rpm"  # RPMS
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    ONE_D_RPM = (  # 1D (identical input to all motors) with RPMs
        "one_d_rpm"
    )
    ONE_D_PID = (  # 1D (identical input to all motors) with PID control
        "one_d_pid"
    )


################################################################################


class ObservationType(Enum):
    """Observation type enumeration class."""

    KIN = (  # Kinematic information (pose, linear and angular velocities)
        "kin"
    )
    RGB = "rgb"  # RGB camera capture in each drone's POV
