import numpy as np


def net_evaluation(sp, act):
    # dice = dice_coef(sp, act)
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = ((2 * tp / (2 * tp) + fp + fn)) * 100
    accuracy = (((tp + tn) / (tp + tn + fp + fn))) * 100
    specificity = ((tn / (tn + fp))) * 100
    precision = ((tp / (tp + fp))) * 100
    NPV = ((tn / (tn + fp))) * 100
    F1_score = (((2 * tp) / (2 * tp + fp + fn))) * 100
    Eval1 = [accuracy, precision, specificity, Dice, NPV, F1_score]
    return Eval1
