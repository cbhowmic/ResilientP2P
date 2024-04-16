import numpy as np

dataset = 'har'
num_workers = 30
attack_type = 'little'      # "sign_flipping"    # byzantine   label_flipping
num_attackers = 0

if num_attackers > 0:
    path0 = "/home/bhowmic/PycharmProjects/P2P_learning/HAR/results_%s/%s_attack" % (dataset, attack_type)
else:
    path0 = "/home/bhowmic/PycharmProjects/P2P_learning/HAR/results_%s/noAttack" % dataset
print('path0:', path0)

all_keys = ["average", "CM", "medoid", "TM", "krum", "loss", "no-coop"]

test_accuracy = {}
for key in all_keys:
    print('key', key)
    try:
        path = "%s/%s_%dagents_%dattacker" % (path0, key, num_workers, num_attackers)
        # print('filename', filename)
        loss_values = np.load("%s/average_train_loss.npy" %path)
        print(f'Aggregation method:', key, 'Training loss (min):', np.min(loss_values))
        print(f'Aggregation method:', key, 'Training loss (final):', loss_values[-1])
        acc_values = np.load("%s/average_test_acc.npy" % path)
        print(f'Aggregation method:', key, 'Testing accuracy (max):', np.max(acc_values))
        print(f'Aggregation method:', key, 'Testing accuracy (final):', acc_values[-1])
    except:
        print('failure')
        pass