from .utils import Namespace

# All names the objects in known_types_list below are also defined as strings
# in the known_types namespace/dict below
known_types_list = [
    "galaxy_shear_xi_plus",
    "galaxy_shear_xi_minus",
    "galaxy_shear_xi_plus_imaginary",
    "galaxy_shear_xi_minus_imaginary",
    "galaxy_shear_ee",
    "galaxy_shear_bb",
    "galaxy_shear_eb",
    "galaxy_density_cl",
    "galaxy_density_w",
    "ggl_gamma_t",
    "ggl_gamma_x",
    "ggl_E",
    "ggl_B",
]


known_types = Namespace(known_types_list)
# These let you do, e.g. known_types.shear_ee == 'shear_ee'