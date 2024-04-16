import numpy as np

dataset = 'mnist'
num_workers = 10
attack_type = 'label_flip'      # "sign_flipping"    # byzantine   label_flipping   empire    little
num_attackers = 3
data = 'niid_pathological'    # niid_pathological

if num_attackers > 0:
    path0 = "/home/bhowmic/PycharmProjects/P2P_learning/MNIST/results_%s_%s/%s_attack" % (dataset, data, attack_type)
else:
    path0 = "/home/bhowmic/PycharmProjects/P2P_learning/MNIST/results_%s_%s/noAttack" % (dataset, data)
print('path0:', path0)

all_keys = ["average", "median", "medoid", "TM", "krum", "loss", "no-coop"]

test_accuracy = {}
for key in all_keys:
    print('key', key)
    try:
        path = "%s/%s_%dagents_%dattacker" % (path0, key, num_workers, num_attackers)
        # print('filename', filename)
        loss_values = np.load("%s/average_train_loss.npy" %path)
        print(f'Aggregation method:', key, 'Training loss:', np.min(loss_values))
        acc_values = np.load("%s/average_test_acc.npy" % path)
        print(f'Aggregation method:', key, 'Testing accuracy:', np.max(acc_values))
    except:
        print('failure')
        pass