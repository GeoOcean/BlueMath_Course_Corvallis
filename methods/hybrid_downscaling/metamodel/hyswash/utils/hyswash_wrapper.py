"""SWASH wrapper for the classic HySwash workflow."""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from bluemath_tk.waves.series import series_TMA
from .swash_wrapper import SwashModelWrapper

FrictionZone = Tuple[float, float, float]

class HySwashModelWrapper(SwashModelWrapper):
    """Classic HySwash wrapper with a case-independent friction profile.

    Friction can be supplied either as ``friction_zones`` or as an existing
    ``friction_file``. It is deliberately kept outside ``metamodel_parameters``
    and ``fixed_parameters``, so it does not become an HySwash input variable.

    Each zone is ``(x_start, x_end, value)`` in profile coordinates. Intervals
    are half-open (``x_start <= x < x_end``), except that a zone ending at the
    final grid coordinate also includes that last point.
    """

    def __init__(
        self,
        *args,
        friction_zones: Optional[Sequence[FrictionZone]] = None,
        friction_file: Optional[Union[str, os.PathLike]] = None,
        default_friction: float = 0.002,
        **kwargs,
    ) -> None:
        if friction_zones is not None and friction_file is not None:
            raise ValueError(
                "Use either friction_zones or friction_file, not both."
            )
        if default_friction < 0:
            raise ValueError("default_friction must be non-negative.")

        super().__init__(*args, **kwargs)
        self.friction_zones = friction_zones
        self.friction_file = (
            Path(friction_file).expanduser().resolve()
            if friction_file is not None
            else None
        )
        self.default_friction = float(default_friction)
        self.friction_profile = self._make_friction_profile()

    def build_case_and_render_files(
        self, case_context: dict, case_dir: str
    ) -> None:
        """Render the classic template under SWASH's expected INPUT.txt name."""
        self.build_case(case_context=case_context, case_dir=case_dir)
        for template_name in self.templates_name:
            output_name = (
                "INPUT.txt" if template_name == "INPUT_classic.txt" else template_name
            )
            try:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_dir, output_name),
                )
            except UnicodeDecodeError:
                self.copy_files(
                    src=os.path.join(self.templates_dir, template_name),
                    dst=os.path.join(case_dir, output_name),
                )

    def _make_friction_profile(self) -> np.ndarray:
        """Load or construct the friction profile and validate its shape."""
        if self.friction_file is not None:
            if not self.friction_file.is_file():
                raise FileNotFoundError(
                    f"Friction file not found: {self.friction_file}"
                )
            friction = np.asarray(np.loadtxt(self.friction_file), dtype=float)
            if friction.ndim != 1:
                raise ValueError("The friction file must contain one column.")
        else:
            friction = np.full(len(self.depth_array), self.default_friction)
            x = np.arange(len(self.depth_array)) * self.fixed_parameters["dxinp"]
            occupied = np.zeros(len(self.depth_array), dtype=bool)

            for zone in self.friction_zones or ():
                if len(zone) != 3:
                    raise ValueError(
                        "Each friction zone must be (x_start, x_end, value)."
                    )
                x_start, x_end, value = map(float, zone)
                if x_start < 0 or x_end <= x_start or value < 0:
                    raise ValueError(f"Invalid friction zone: {zone}")
                mask = (x >= x_start) & (x < x_end)
                if np.isclose(x_end, x[-1]):
                    mask |= np.isclose(x, x[-1])
                if not mask.any():
                    raise ValueError(f"Friction zone is outside the grid: {zone}")
                if np.any(occupied & mask):
                    raise ValueError(f"Friction zones overlap at: {zone}")
                friction[mask] = value
                occupied |= mask

        if len(friction) != len(self.depth_array):
            raise ValueError(
                "The friction profile and depth_array must have the same length "
                f"({len(friction)} != {len(self.depth_array)})."
            )
        if not np.all(np.isfinite(friction)) or np.any(friction < 0):
            raise ValueError("Friction values must be finite and non-negative.")
        return friction

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """Create the wave boundary and fixed friction files for one case."""
        super().build_case(case_context=case_context, case_dir=case_dir)
        waves = series_TMA(
            waves=case_context["waves_dict"], depth=self.depth_array[0]
        )
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )
        np.savetxt(
            os.path.join(case_dir, "friction.txt"),
            self.friction_profile,
            fmt="%.6f",
        )
