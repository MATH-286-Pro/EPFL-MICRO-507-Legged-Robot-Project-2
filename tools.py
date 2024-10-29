import numpy as np
from numpy import sin,cos


# compute relative joint angle
# 计算相对关节角度
def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    # [TODO]
    J[0, 0] = -l1*cos(q[0])-l2*cos(q[0]+q[1]) ###
    J[1, 0] = +l1*sin(q[0])+l2*sin(q[0]+q[1]) ###
    J[0, 1] = -l2*cos(q[0]+q[1]) ###
    J[1, 1] = +l2*sin(q[0]+q[1]) ###

    # foot pos
    pos = np.zeros(2)
    # [TODO]
    pos[0] = -l1*sin(q[0])-l2*sin(q[0]+q[1]) ###
    pos[1] = -l1*cos(q[0])-l2*cos(q[0]+q[1]) ###

    return J, pos