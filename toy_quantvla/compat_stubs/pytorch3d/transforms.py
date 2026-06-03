"""Fail-fast pytorch3d.transforms stub for GR00T LIBERO smoke tests."""


def _not_available(*args, **kwargs):
    raise NotImplementedError(
        "pytorch3d.transforms is stubbed for LIBERO smoke tests only; "
        "install real pytorch3d for rotation conversion."
    )


axis_angle_to_matrix = _not_available
matrix_to_axis_angle = _not_available
euler_angles_to_matrix = _not_available
matrix_to_euler_angles = _not_available
quaternion_to_matrix = _not_available
matrix_to_quaternion = _not_available
rotation_6d_to_matrix = _not_available
matrix_to_rotation_6d = _not_available
