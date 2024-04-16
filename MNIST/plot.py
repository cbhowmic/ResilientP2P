import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from copy import deepcopy

plt.style.use("bmh")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 155
plt.rcParams['axes.facecolor'] = 'white'

def get_label(key):
    if key == 'loss':
        label = 'AdapAgg'
    elif key == 'median':
        label = 'CM'
    else:
        label = key
    return label

def plot_subfigures(path0):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    if num_attackers == 0:
        path = "%s/noAttack" % (path0)
    else:
        path = "%s/%s_attack" % (path0, attack_type)
    mean_train_loss, max_train_loss, min_train_loss = {}, {}, {}
    mean_test_acc, max_test_acc, min_test_acc = {}, {}, {}
    # """ Trial"""
    # for key in ["average"]:
    #     if num_attackers > 0:
    #         normalAgents = np.load("%s/%s_%dagents_%dattacker/normalAgents.npy" % (path, key, num_workers, num_attackers))
    #     else:
    #         normalAgents = np.arange(0, num_workers)
    #     mean_train_loss[key] = np.load("%s/%s_%dagents_%dattacker/average_train_loss.npy" % (path, key, num_workers, num_attackers))
    #     all_train_loss = np.transpose(np.load("%s/%s_%dagents_%dattacker/individual_average_train_loss.npy" % (
    #         path, key, num_workers, num_attackers)))
    #     max_train_loss[key] = np.max(all_train_loss[normalAgents], 0)
    #     min_train_loss[key] = np.min(all_train_loss[normalAgents], 0)
    #     mean_test_acc[key] = np.load(
    #         "%s/%s_%dagents_%dattacker/average_test_acc.npy" % (path, key, num_workers, num_attackers))
    #     all_test_acc = np.transpose(np.load("%s/%s_%dagents_%dattacker/individual_average_test_acc.npy" % (
    #         path, key, num_workers, num_attackers)))
    #     max_test_acc[key] = np.max(all_test_acc[normalAgents], 0)
    #     min_test_acc[key] = np.min(all_test_acc[normalAgents], 0)
    # print('loading done')
    # for key in ["average"]:
    #     plt.plot(mean_train_loss[key], label=key)
    #     print('hfshfsf')
    #     plt.fill_between(min_train_loss[key], max_train_loss[key], alpha=0.3)
    #     plt.xlabel(r'Time steps')
    #     plt.ylabel(r'Train loss')
    #     plt.tight_layout()
    #     plt.show()
    # fig.savefig('/home/bhowmic/PycharmProjects/P2P_learning/MNIST/results.png')
    # """Trial end"""

    # mean_train_loss, max_train_loss, min_train_loss = {}, {}, {}
    # for key" in ["average"]:
    #     mean_train_loss[key] = np.load("%s/%s_%dagents_%dattacker/average_train_loss.npy" % (path, key, num_workers, num_attackers))
    #     if num_attackers > 0:
    #         normalAgents = np.load("%s/%s_%dagents_%dattacker/normalAgents.npy" % (path, key, num_workers, num_attackers))
    #     else:
    #         normalAgents = np.arange(0, num_workers)
    #     all_train_loss = np.transpose(np.load("%s/%s_%dagents_%dattacker/individual_average_train_loss.npy" % (path, key, num_workers, num_attackers)))
    #     max_train_loss[key] = np.max(all_train_loss[normalAgents], 0)
    #     print('rule', key, 'size', np.shape(mean_train_loss[key]))
    #     print('normal', normalAgents)
    #     print('all loss', np.shape(all_train_loss[normalAgents]))
    #     print('max loss', np.shape(max_train_loss[key]))


    try:
        for key in ["no-coop", "medoid", "average", "loss", "krum", "mKrum", "CM", "TM", "median"]:  #,
            try:
                if num_attackers > 0:
                    normalAgents = np.load("%s/%s_%dagents_%dattacker/normalAgents.npy" % (path, key, num_workers, num_attackers))
                else:
                    normalAgents = np.arange(0, num_workers)
                mean_train_loss[key] = np.load("%s/%s_%dagents_%dattacker/average_train_loss.npy" % (path, key, num_workers, num_attackers))
                all_train_loss = np.transpose(np.load("%s/%s_%dagents_%dattacker/individual_average_train_loss.npy" % (
                path, key, num_workers, num_attackers)))
                max_train_loss[key] = np.max(all_train_loss[normalAgents], 0)
                min_train_loss[key] = np.min(all_train_loss[normalAgents], 0)
                mean_test_acc[key] = np.load("%s/%s_%dagents_%dattacker/average_test_acc.npy" % (path, key, num_workers, num_attackers))
                all_test_acc = np.transpose(np.load("%s/%s_%dagents_%dattacker/individual_average_test_acc.npy" % (
                path, key, num_workers, num_attackers)))
                max_test_acc[key] = np.max(all_test_acc[normalAgents], 0)
                min_test_acc[key] = np.min(all_test_acc[normalAgents], 0)
                print(key, np.shape(mean_train_loss[key]), np.shape(mean_test_acc[key]))

    #             for x in range(0, len(steps[key]) * interval, interval):
    #                 if x < span:
    #                     mean_reward_copy[key][x] = np.mean(mean_reward[key][:x + span])
    #                 elif x >= span and x < len(steps[key]) - span:
    #                     mean_reward_copy[key][x] = np.mean(mean_reward[key][x - span:x + span])
    #                 else:
    #                     mean_reward_copy[key][x] = np.mean(mean_reward[key][x - span:len(steps[key])])
    #
    #             for x in range(0, len(steps[key]) * interval, interval):
    #                 if x < span:
    #                     max_reward_copy[key][x] = np.mean(max_reward[key][:x + span])
    #                 elif x >= span and x < len(steps[key]) - span:
    #                     max_reward_copy[key][x] = np.mean(max_reward[key][x - span:x + span])
    #                 else:
    #                     max_reward_copy[key][x] = np.mean(max_reward[key][x - span:len(steps[key])])
    #
    #             for x in range(0, len(steps[key]) * interval, interval):
    #                 if x < span:
    #                     min_reward_copy[key][x] = np.mean(min_reward[key][:x + span])
    #                 elif x >= span and x < len(steps[key]) - span:
    #                     min_reward_copy[key][x] = np.mean(min_reward[key][x - span:x + span])
    #                 else:
    #                     min_reward_copy[key][x] = np.mean(min_reward[key][x - span:len(steps[key])])
    #
            except:
                pass
        line_style = {"no-coop": "-", "average": "-", "medoid": "-", "loss": "-", "krum": "-", "mKrum": "-", "CM": "--", "median": "--", "TM": "-"}
        fig = plt.figure(figsize=(10, 6.5))
        for key in ["average", "CM", "median", "medoid", "TM", "krum", "loss"]:    #  "no-coop", "TM", "krum", "mKrum", "CM",
            try:
                label = get_label(key)
                loss_interval = max_train_loss[key][::interval]
                x_values = np.arange(0, len(max_train_loss[key]), interval)
                plt.plot(x_values, loss_interval, label=label, linestyle=line_style[key], linewidth=2)
                # plt.plot(max_train_loss[key], label=label, linestyle=line_style[key], linewidth=2)
                # plt.plot(mean_train_loss[key], label=key, linestyle=line_style[key])
                # plt.fill_between(min_train_loss[key], max_train_loss[key], alpha=0.3)
            except:
                pass
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Train loss')
        plt.grid(False)
        plt.tight_layout()
        plt.xlim([0, max_steps])
        # plt.ylim([0, 1])
        plt.legend()
        plt.show()
        if num_attackers > 0:
            fig.savefig('%s/figures/train_loss_%dworkers_%dattackers_%s.png' % (path0, num_workers, num_attackers, attack_type))
        else:
            fig.savefig('%s/figures/train_loss_%dworkers_noAttack.png' % (path0, num_workers))

        fig = plt.figure(figsize=(10, 6.5))
        for key in ["average", "CM", "median", "medoid", "TM", "krum", "loss"]:    #  "no-coop", "TM", "krum", "mKrum", "CM",
            try:
                label = get_label(key)
                acc_interval = min_test_acc[key][::interval]
                x_values = np.arange(0, len(min_test_acc[key]), interval)
                plt.plot(x_values, acc_interval, label=label, linestyle=line_style[key], linewidth=2)
                # plt.plot(min_test_acc[key], label=label, linestyle=line_style[key], linewidth=2)
                # plt.plot(mean_test_acc[key], label=key, linestyle=line_style[key])
                # plt.fill_between(min_test_acc[key], max_test_acc[key], alpha=0.3)
            except:
                pass
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Test accuracy')
        plt.grid(False)
        plt.tight_layout()
        plt.xlim([0, max_steps])
        # plt.ylim([0, 1])
        plt.legend()
        plt.show()
        if num_attackers > 0:
            fig.savefig('%s/figures/test_acc_%dworkers_%dattackers_%s.png' % (path0, num_workers, num_attackers, attack_type))
        else:
            fig.savefig('%s/figures/test_acc_%dworkers_noAttack.png' % (path0, num_workers))
    except:
        pass




if __name__ == '__main__':
    dataset ='mnist'
    num_workers = 10
    num_attackers = 3
    attack_type = "empire"        # "arbitrary"  # "sign_flip"  "label_flip" "little" "empire"
    max_steps = 100
    interval = 1
    data_distribution = 'niid_pathological'     # niid_pathological
    network = 'conv'
    path = "/home/bhowmic/PycharmProjects/P2P_learning/MNIST/results_%s_%s_%s" % (dataset, data_distribution, network)
    plot_subfigures(path)