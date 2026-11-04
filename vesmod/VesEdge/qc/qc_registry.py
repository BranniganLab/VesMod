from ..experimental.area_qc import AreaDeviationCheck
from ..experimental.minimum_radius_qc import MinimumRadiusCheck
from ..experimental.singleton_qc import SingletonDeviationCheck
from .checks.edge_filtering import CurvatureCheck
from .checks.localized_deviation_qc import LocalizedDeviationCheck
from .contract import QCCheck


QC_CHECKS: tuple[QCCheck, ...] = (
    CurvatureCheck(),
    MinimumRadiusCheck(),
    LocalizedDeviationCheck(),
    SingletonDeviationCheck(),
    AreaDeviationCheck(),
)
