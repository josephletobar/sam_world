FX = None
FY = None
CX = None
CY = None
DEPTH_SCALE = None


def configure_camera_intrinsics(fx=None, fy=None, cx=None, cy=None, depth_scale=None):
    global FX, FY, CX, CY, DEPTH_SCALE

    if fx is not None:
        FX = float(fx)
    if fy is not None:
        FY = float(fy)
    if cx is not None:
        CX = float(cx)
    if cy is not None:
        CY = float(cy)
    if depth_scale is not None:
        DEPTH_SCALE = float(depth_scale)


def require_camera_configured():
    if any(value is None for value in (FX, FY, CX, CY, DEPTH_SCALE)):
        raise RuntimeError(
            "Camera intrinsics are not configured. "
            "Call load_data(source) with a dataset containing camera_intrinsics.txt first."
        )
