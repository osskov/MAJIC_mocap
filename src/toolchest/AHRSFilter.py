from typing import List, Dict, Callable, Any, Optional
import ahrs.filters as filt
import numpy as np
from scipy.spatial.transform import Rotation


ALL_FILTERS = {
    "Angular Rate": [filt.AngularRate, True, False, False, [0, 0, 0], [0, 0, 0]],
                   "Ekf": [filt.EKF, True, True, True, [0, 0, -1], [1, 1, 1]],
                   "Madgwick": [filt.Madgwick, True, True, None, [0, 0, 1], [1, 0, 1]],
                   "Mahony": [filt.Mahony, True, True, None, [0, 0, 1], [-1, 0, 1]],
                   }


class AHRSFilter:
    """ Initializes, stores, and updates multiple AHRS filters. """

    last_q: Rotation = Rotation.identity()
    filter: any
    R_wa: np.ndarray = np.eye(3)
    q_world_to_ahrs: Rotation = Rotation.identity()

    def __init__(self,
                 select_filter: str = 'Mahony'):
        if select_filter in ALL_FILTERS.keys():
            self.filter_type = select_filter
            filter_call, self.gyro_req, self.acc_req, self.mag_req, self.a_ref, self.m_ref = ALL_FILTERS[select_filter]
            if select_filter == 'Madgwick':
                self.filter = filter_call(beta=1)
            elif select_filter == 'Ekf':
                self.filter = filter_call(noises=[0.01, 0.05, 0.05])
                self.filter.mag = True
            elif select_filter == 'Mahony':
                self.filter = filter_call(Kp=1.0, Ki=0.0)
            else:
                self.filter = filter_call()
        else:
            raise ValueError(f"Warning! {select_filter} isn't a valid filter type.")

        self.R_wa = np.eye(3)
        self.q_world_to_ahrs = Rotation.from_matrix(self.R_wa)
        self.last_q = Rotation.identity()

    def update(self, dt: float, acc: np.ndarray, gyro: np.ndarray, mag: Optional[np.ndarray] = None):
        if hasattr(self.filter, 'frequency'):
            self.filter.frequency = 1 / dt
        if hasattr(self.filter, 'Dt'):
            self.filter.Dt = dt
        if hasattr(self.filter, 'dt'):
            self.filter.dt = dt

        q_current = np.roll(self.last_q.as_quat(), shift=1)  # Convert to AHRS [w, x, y, z]

        if self.mag_req is False:
            updated_q = self.filter.update(q_current, gyro, 'series')
        elif self.mag_req is True:
            updated_q = self.filter.update(q_current, gyr=gyro, acc=acc, mag=mag)
        else:
            if mag is None:
                updated_q = self.filter.updateIMU(q_current, gyr=gyro, acc=acc)
            else:
                updated_q = self.filter.updateMARG(q=q_current, gyr=gyro, acc=acc, mag=mag)

        self.set_last_q(updated_q)

    def get_last_q(self) -> np.ndarray:
        """ Convert SciPy `[x, y, z, w]` to AHRS `[w, x, y, z]`. """
        return np.roll(self.last_q.as_quat(), shift=1)

    def get_last_R(self) -> np.ndarray:
        return self.last_q.as_matrix()

    def set_last_q(self, q: np.ndarray):
        """ Convert AHRS `[w, x, y, z]` to SciPy `[x, y, z, w]` and store it. """
        self.last_q = Rotation.from_quat(q[[1, 2, 3, 0]])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filter_type": self.filter_type
        }

    @staticmethod
    def from_dict(ahrs_dict: Dict) -> 'AHRSFilter':
        ahrs = AHRSFilter(select_filter=ahrs_dict["filter_type"])
        return ahrs