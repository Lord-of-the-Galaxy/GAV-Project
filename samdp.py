import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from cuml import TSNE

from sklearn.cluster import MiniBatchKMeans
from scipy.stats import entropy

from time import time

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

def kmeans_st(coords, values, n_clusters, window_size=4, window_weight_falloff=0.75):
    vecs = np.ndarray((coords.shape[0], window_size*4 + 3))
    for i in range(window_size*2+1):
        vecs[:,2*i] = (window_weight_falloff**(abs(i-window_size)))*np.roll(coords, i-window_size, axis=0)[:,0]
        vecs[:,2*i  + 1] = (window_weight_falloff**(abs(i-window_size)))*np.roll(coords, i-window_size, axis=0)[:,1]
    vecs[:,window_size*4 + 2] = values
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(time()), n_init=3).fit(vecs)
    #print(kmeans.inertia_)
    #print(kmeans.cluster_centers_[:, window_size*4 + 2])
    #print(kmeans.cluster_centers_[:, window_size*2:window_size*2 + 2])
    return kmeans.cluster_centers_[:, window_size*2:window_size*2 + 2], kmeans.cluster_centers_[:, window_size*4 + 2], kmeans.labels_, kmeans.inertia_


def compute_transition_matrix(n_clusters, labels, min_skill_length):
    T = np.zeros((n_clusters, n_clusters))
    for i, l in enumerate(labels):
        if i == len(labels)-1:
            continue
        elif labels[i] != labels[i+1]:
            if np.all(labels[i+1:i+1+min_skill_length] == labels[i+1]):
                T[labels[i], labels[i+1]] += 1
    return T

def compute_prob_matrix(transition_matrix, threshold):
    P = np.copy(transition_matrix)
    P = P/P.sum(axis=1)[:, np.newaxis]
    P[np.isnan(P)] = 0
    P[P < threshold] = 0
    P = P/P.sum(axis=1)[:, np.newaxis]
    P[np.isnan(P)] = 0
    return P

def calc_entropy(P, cluster_count):
    e = entropy(P.T)
    e_finite_ind = np.isfinite(e)
    ent = np.average(a=e[e_finite_ind],weights=cluster_count[e_finite_ind])
    return ent

def calc_v_dqn(n_clusters, labels, values):
    v_dqn = np.zeros(n_clusters)
    for i in range(n_clusters):
        v_dqn[i] = np.mean(values[labels == i])
    return v_dqn

def calc_v_samdp(n_clusters, P, skill_lengths, samdp_rewards, gamma=0.99):
    GAMMA = np.diag(gamma**skill_lengths)
    return np.linalg.pinv(np.identity(n_clusters) - GAMMA@P)@samdp_rewards

def calc_samdp_rewards(n_clusters, labels, rewards, min_skill_length, gamma=0.99):
    mean_rewards = np.zeros(n_clusters)
    mean_lengths = np.zeros(n_clusters)
    num_skills = np.zeros(n_clusters)
    
    t = 0
    cum_reward = 0
    prev_label = labels[0]
    for i in range(1, len(labels)):
        cum_reward += (gamma**t)*rewards[i]
        if labels[i] == prev_label:
            t += 1
        else:
            if t >= min_skill_length:
                mean_rewards[prev_label] += cum_reward
                mean_lengths[prev_label] += t
                num_skills[prev_label] += 1
            prev_label = labels[i]
            cum_reward = 0
            t = 0
    num_skills[num_skills == 0] = 1
    mean_rewards = mean_rewards/num_skills
    mean_lengths = mean_lengths/num_skills
    
    return mean_rewards, mean_lengths

def samdp(activation_data, values, rewards, n_clusters=20, window_size=4, min_skill_length=4, window_weight_falloff=0.75, threshold=0.01, gamma=0.99):
    coords, cluster_values, labels, inertia = kmeans_st(activation_data, values, n_clusters, window_size, window_weight_falloff)
    T = compute_transition_matrix(n_clusters, labels, min_skill_length)
    P = compute_prob_matrix(T, threshold)
    samdp_rewards, skill_lengths = calc_samdp_rewards(n_clusters, labels, rewards, min_skill_length, gamma)
    v_samdp = calc_v_samdp(n_clusters, P, skill_lengths, samdp_rewards, gamma)
    v_dqn = calc_v_dqn(n_clusters, labels, values)
    vmse = np.linalg.norm(v_samdp - v_dqn)/np.linalg.norm(v_dqn)
    e = calc_entropy(P, np.bincount(labels))
    return coords, v_samdp, labels, P, vmse, inertia, e


def samdp_search(activation_data, values, rewards, min_clusters, max_clusters, min_window_size, max_window_size, min_skill_length=4, window_weight_falloff=0.75, threshold=0.01, gamma=0.99):
    vmse = np.zeros((max_clusters - min_clusters + 1)*(max_window_size - min_window_size + 1))
    inertia = np.zeros((max_clusters - min_clusters + 1)*(max_window_size - min_window_size + 1))
    entropy = np.zeros((max_clusters - min_clusters + 1)*(max_window_size - min_window_size + 1))
    
    coords = []
    v_samdp = []
    labels = []
    P = []
    
    for i in range(min_clusters, max_clusters+1):
        for j in range(min_window_size, max_window_size+1):
            print(f"Computing {i}, {j}")
            coords_, v_samdp_, labels_, P_, vmse[(i-min_clusters)*(max_window_size - min_window_size + 1) + j-min_window_size], inertia[(i-min_clusters)*(max_window_size - min_window_size + 1) + j-min_window_size], entropy[(i-min_clusters)*(max_window_size - min_window_size + 1) + j-min_window_size] = samdp(activation_data, values, rewards, i, j, min_skill_length, window_weight_falloff, threshold, gamma)
            coords.append(coords_)
            v_samdp.append(v_samdp_)
            labels.append(labels_)
            P.append(P_)
    vmse_srt = np.argsort(vmse, axis=None)
    inertia_srt = np.argsort(inertia, axis=None)
    entropy_srt = np.argsort(entropy, axis=None)
    for p in range(vmse.shape[0]):
        vmse_min = vmse_srt[:p]
        inertia_min = inertia_srt[:p]
        entropy_min = entropy_srt[:p]
        candidates = set(vmse_min).intersection(set(inertia_min)).intersection(set(entropy_min))
        if candidates:
            print("Found candidate at p =", p)
            n = candidates.pop()
            n_clusters, window_size = n//(max_window_size - min_window_size + 1) + min_clusters, n%(max_window_size - min_window_size + 1) + min_window_size
            print("Clusters:", n_clusters, "Window size:", window_size)
            return n_clusters, coords[n], v_samdp[n], labels[n], P[n], vmse[n], inertia[n], entropy[n]


def main():
    game = 'breakout'
    activation_data = np.load(game+'/activations_0.npy')[:25000]
    qvalues = np.load(game+'/qvalues_0.npy')[:25000]
    rewards = np.load(game+'/rewards_0.npy')[:25000]
    values = np.max(qvalues, axis=1)

    tsne = TSNE(n_components=2, perplexity=120, n_neighbors=600, verbose=1, random_state=1534,method='fft')
    print("Running tSNE")
    data_t = tsne.fit_transform(activation_data)
    print("tSNE done")
    
    plt.scatter(data_t[:,0], data_t[:,1], s=4, c=values, cmap='gist_rainbow')

    cbar = plt.colorbar()
    cbar.set_label('Estimated Value')

    plt.show()

    print("Running SAMDP search")

    nc, coords, v_samdp, labels, P, vmse, inertia, entropy = samdp_search(data_t, values, rewards, 15, 25, 2, 5)

    # nc = 15

    # coords, cluster_values, labels, inertia = kmeans_st(data_t, values, nc)

    # P = compute_prob_matrix(compute_transition_matrix(nc, labels, 4), 0.01)

    # samdp_rewards, skill_lengths = calc_samdp_rewards(nc, labels, rewards, 4)

    # v_samdp = calc_v_samdp(nc, P, skill_lengths, samdp_rewards)
    # v_dqn = calc_v_dqn(nc, labels, values)

    # print(np.linalg.norm(v_samdp - v_dqn)/np.linalg.norm(v_dqn))

    plt.scatter(data_t[:,0], data_t[:,1], s=4, c=values, cmap='gist_rainbow')

    cbar = plt.colorbar()
    cbar.set_label('Estimated Value')

    for i in range(nc):
        for j in range(nc):
            if P[i, j] > 0.05:
                #arrow = FancyArrow(coords[i, 0], coords[i, 1], coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1], width=0.1*P[i, j], head_width=0.5*P[i, j], head_length=None, overhang=0.7, color='black', alpha=0.75)
                #plt.gca().add_patch(arrow)
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='black', linewidth=P[i, j]*5, alpha=0.5)

    plt.scatter(coords[:, 0], coords[:, 1], s=100, c='black', marker='D')

    plt.show()
    

if __name__ == '__main__':
    main()