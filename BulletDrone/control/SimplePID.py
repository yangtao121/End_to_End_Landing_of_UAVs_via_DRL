import numpy as np

from BulletDrone.utils.ControlFunction import sat_gd


class PIDParam:
    def __init__(self, Kph, Kpz, KVhp, KVzp, KVhi, KVhd, KVzi, KVzd, KOmega_x, KOmega_y, KOmega_z, Kwp_x, Kwp_y, Kwp_z,
                 Kwi_x, Kwi_y, Kwi_z, Kwd_x, Kwd_y, Kwd_z,
                 acc_hmax, acc_zmax,
                 delta_hmax, delta_zmax,
                 m, inertia):
        self.Kp = np.array([Kph, Kph, Kpz])
        self.KVp = np.array([KVhp, KVhp, KVzp])
        self.KVi = np.array([KVhi, KVhi, KVzi])
        self.KVd = np.array([KVhd, KVhd, KVzd])

        self.KOmega = np.array([KOmega_x, KOmega_y, KOmega_z])
        self.Kwp = np.array([Kwp_x, Kwp_y, Kwp_z])
        self.Kwi = np.array([Kwi_x, Kwi_y, Kwi_z])
        self.Kwd = np.array([Kwd_x, Kwd_y, Kwd_z])

        self.acc_hmax = acc_hmax
        self.acc_zmax = acc_zmax
        self.delta_hmax = delta_hmax
        self.delta_zmax = delta_zmax

        self.m = m
        self.inertia = np.array([
            [inertia[0], 0, 0],
            [0, inertia[1], 0],
            [0, 0, inertia[2]]
        ])


class SimplePIDControl:

    def __init__(self, inverse_M, param: PIDParam, g=9.8):
        self.param = param
        self.inverse_M = inverse_M

        self.last_pos = np.zeros(3)
        self.integral_eV = np.zeros(3)
        self.last_eV = np.zeros(3)

        self.last_Euler = np.zeros(3)
        self.integral_e_omega = np.zeros(3)
        self.last_e_omega = np.zeros(3)

        self.G = np.array([0, 0, -g])

    def PositionControl(self,
                        cur_pos,
                        cur_vel,
                        target_pos,
                        target_yaw,
                        control_time_step,
                        ):
        pos_ed = target_pos - cur_pos
        Vd = pos_ed * self.param.Kp

        V = cur_vel
        eV = V - Vd
        # eVh = sat_gd(eV[:2], self.param.delta_hmax)
        # eVz = sat_gd(eV[2:], self.param.delta_zmax)
        # eV = np.array([eVh[0], eVh[1], eVz[0]])

        dot_eV = (eV - self.last_eV) / control_time_step

        self.integral_eV = self.integral_eV + eV * control_time_step

        ad = -self.param.KVp * eV - self.param.KVi * self.integral_eV - self.param.KVd * dot_eV
        # adh = sat_gd(ad[:2], self.param.acc_hmax)
        # adz = sat_gd(ad[2:], self.param.acc_zmax)
        # ad = np.array([adh[0], adh[1], adz[0]])

        r3_d = (ad - self.G) / np.linalg.norm(ad - self.G, ord=2)

        pitch_x = np.cos(target_yaw) * r3_d[0] + np.sin(target_yaw) * r3_d[1]
        sign_pitch = np.sign(r3_d[2])
        if sign_pitch == 0:
            sign_pitch = 1

        pitch = np.arctan2(sign_pitch * pitch_x, sign_pitch * r3_d[2])
        # pitch = np.arctan(r3_d[2]/pitch_x)
        # pitch = np.clip(pitch, -0.5, 0.5)
        # print(r3_d[2]/pitch_x)
        # print('({},{})'.format(pitch_x, r3_d[2]))
        roll_ = np.sin(target_yaw) * r3_d[0] - np.cos(target_yaw) * r3_d[1]
        # print(roll_)
        roll = np.arcsin(roll_)
        # roll = np.clip(roll, -0.5, 0.5)
        # print(r3_d)
        # print(roll)

        # if sign_pitch < 0:
        #     roll = roll - np.sign(pitch) * np.pi

        r3_d = r3_d.reshape((3, 1))
        ad_G = (ad - self.G).reshape((3, 1))
        fd = np.squeeze(self.param.m * np.transpose(r3_d) @ ad_G)

        target = np.array([roll, pitch, target_yaw, fd])
        print(target)

        self.last_pos = cur_pos
        self.last_eV = eV

        return target

    def AttitudeControl(self,
                        cur_Euler,
                        cur_omega,
                        target,
                        control_time_step
                        ):
        d_Euler = target[:3]
        fd = target[3]
        e_Euler = cur_Euler - d_Euler
        omega = cur_omega
        omega_d = -self.param.KOmega * e_Euler
        e_omega = omega - omega_d

        self.integral_e_omega = self.integral_e_omega + e_omega * control_time_step

        dot_e_omega = (e_omega - self.last_e_omega) / control_time_step

        tau_d = -self.param.Kwp * e_omega - self.param.Kwi * self.integral_e_omega - self.param.Kwd * dot_e_omega

        target = np.array([fd, tau_d[0], tau_d[1], tau_d[2]])

        self.last_Euler = cur_Euler
        self.last_e_omega = e_omega

        return target

    # def AttitudeControl2(self,
    #                      cur_rotation_matrix,
    #                      cur_omega,
    #                      target,
    #                      control_time_step
    #                      ):

    def ControlDistribution(self, target):
        target = target.reshape((4, 1))
        RPM = self.inverse_M @ target
        RPM = np.sqrt(np.squeeze(RPM.T, axis=0))
        return RPM

    def reset(self):
        self.last_pos = np.zeros(3)
        self.integral_eV = np.zeros(3)
        self.last_eV = np.zeros(3)

        self.last_Euler = np.zeros(3)
        self.integral_e_omega = np.zeros(3)
        self.last_e_omega = np.zeros(3)

    def VelocityControl(self,
                        cur_vel,
                        target_vel,
                        target_yaw,
                        control_time_step,
                        ):
        Vd = target_vel
        V = cur_vel
        eV = V - Vd
        # eVh = sat_gd(eV[:2], self.param.delta_hmax)
        # eVz = sat_gd(eV[2:], self.param.delta_zmax)
        # eV = np.array([eVh[0], eVh[1], eVz[0]])

        dot_eV = (eV - self.last_eV) / control_time_step

        self.integral_eV = self.integral_eV + eV * control_time_step

        ad = -self.param.KVp * eV - self.param.KVi * self.integral_eV - self.param.KVd * dot_eV
        # adh = sat_gd(ad[:2], self.param.acc_hmax)
        # adz = sat_gd(ad[2:], self.param.acc_zmax)
        # ad = np.array([adh[0], adh[1], adz[0]])

        r3_d = (ad - self.G) / np.linalg.norm(ad - self.G, ord=2)

        pitch_x = np.cos(target_yaw) * r3_d[0] + np.sin(target_yaw) * r3_d[1]
        sign_pitch = np.sign(r3_d[2])
        if sign_pitch == 0:
            sign_pitch = 1

        pitch = np.arctan2(sign_pitch * pitch_x, sign_pitch * r3_d[2])
        # pitch = np.arctan(r3_d[2]/pitch_x)
        # pitch = np.clip(pitch, -0.5, 0.5)
        # print(r3_d[2]/pitch_x)
        # print('({},{})'.format(pitch_x, r3_d[2]))
        roll_ = np.sin(target_yaw) * r3_d[0] - np.cos(target_yaw) * r3_d[1]
        # print(roll_)
        roll = np.arcsin(roll_)
        # roll = np.clip(roll, -0.5, 0.5)
        # print(r3_d)
        # print(roll)

        # if sign_pitch < 0:
        #     roll = roll - np.sign(pitch) * np.pi

        r3_d = r3_d.reshape((3, 1))
        ad_G = (ad - self.G).reshape((3, 1))
        fd = np.squeeze(self.param.m * np.transpose(r3_d) @ ad_G)

        target = np.array([roll, pitch, target_yaw, fd])
        # print(target)

        self.last_eV = eV

        return target
