import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gpd.gym_pybullet_drones.utils.enums import (
    DroneModel,
)
drone_model = DroneModel.CF2X
DRONE_MODEL = drone_model
URDF = DRONE_MODEL.value + ".urdf"
def _parseURDFParameters():
    """Loads parameters from an URDF file.

    This method is nothing more than a custom XML parser for the .urdf
    files in folder `assets/`.

    """
    URDF_TREE = etxml.parse(
        pkg_resources.resource_filename(
            "gpd.gym_pybullet_drones", "assets/" + 'cf2x.urdf'
        )
    ).getroot()
    M = float(URDF_TREE[1][0][1].attrib["value"])
    L = float(URDF_TREE[0].attrib["arm"])
    THRUST2WEIGHT_RATIO = float(
        URDF_TREE[0].attrib["thrust2weight"]
    )
    IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
    IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
    IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
    J = np.diag([IXX, IYY, IZZ])
    J_INV = np.linalg.inv(J)
    KF = float(URDF_TREE[0].attrib["kf"])
    KM = float(URDF_TREE[0].attrib["km"])
    COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
    COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
    COLLISION_SHAPE_OFFSETS = [
        float(s)
        for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")
    ]
    COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
    MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
    GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
    PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
    DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
    DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
    DRAG_COEFF = np.array(
        [DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z]
    )
    DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
    DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
    DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])
    return (
        M,
        L,
        THRUST2WEIGHT_RATIO,
        J,
        J_INV,
        KF,
        KM,
        COLLISION_H,
        COLLISION_R,
        COLLISION_Z_OFFSET,
        MAX_SPEED_KMH,
        GND_EFF_COEFF,
        PROP_RADIUS,
        DRAG_COEFF,
        DW_COEFF_1,
        DW_COEFF_2,
        DW_COEFF_3,
    )

G = 9.8
        #### Load the drone properties from the .urdf file #########
(
    M,
    L,
    THRUST2WEIGHT_RATIO,
    J,
    J_INV,
    KF,
    KM,
    COLLISION_H,
    COLLISION_R,
    COLLISION_Z_OFFSET,
    MAX_SPEED_KMH,
    GND_EFF_COEFF,
    PROP_RADIUS,
    DRAG_COEFF,
    DW_COEFF_1,
    DW_COEFF_2,
    DW_COEFF_3,
) = _parseURDFParameters()
GRAVITY = G * M
HOVER_RPM = np.sqrt(GRAVITY / (4 * KF))
max_rpm = np.sqrt(
            (THRUST2WEIGHT_RATIO * GRAVITY) / (4 * KF)
        )



