from skimage.morphology import skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    cldice = []
    for b in range(v_p.shape[0]):
        v_p = v_p[b,0,:,:,:].cpu().numpy()
        v_l = v_l[b,0,:,:,:].cpu().numpy()
        """[this function computes the cldice metric]

        Args:
            v_p ([bool]): [predicted image]
            v_l ([bool]): [ground truth image]

        Returns:
            [float]: [cldice metric]
        """
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))

        cldice.append(2*tprec*tsens/(tprec+tsens))

    return np.mean(np.array(cldice))