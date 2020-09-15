import csv
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import pickle
import numpy as np
import glob

from pathlib import Path

import os

DIR = './AER/scores/'


###
# Utility functions to generate plots
###


def get_all_score_filenames_in_folder(root):
    filenames = []
    for fn in glob.iglob(root + "**/*.p", recursive=True):
        filenames.append(os.path.relpath(fn, DIR))
    return filenames

# WIP
def get_model_training_errors(folder):
    losses = dict()
    rmses = dict()
    DIR = './AER/model_training/'
    for filename in os.listdir(DIR+folder+'/'):
        obj = pickle.load(open(DIR+folder+'/'+filename, 'rb'))
        if True:#len(obj) > 150:#or len(obj) == 100:
            param = filename.split('h')[1].split('_')[0]
            if filename.startswith('loss'):
                losses[param] = obj
            else:
                rmses[param] = obj
    return losses, rmses

# WIP
def plot_training_errors(folder):
    losses, rmses = get_model_training_errors(folder)

    keys = losses.keys()

    depths = [0,1,2,3,4,6,8]

    for i in depths:
        tmpkeys=[]
        for key in keys:
            if not key.startswith(str(i)):
                continue
            tmpkeys.append(key)
            plt.plot(losses[key])

            print(key + ': ' + str(losses[key][-1]))

        plt.legend(tmpkeys)
        # plt.show()

    # for key in keys:
    #     plt.plot(rmses[key])
    # plt.legend(keys)
    # plt.show()

def get_progress(iteration):
    curr_max = -200
    progress = []
    for row in iteration:
        if row > curr_max:
            curr_max = row
        progress.append(curr_max)
    return progress

def plot_scores_by_rollout(env = 'MountainCar', num_timesteps = '300000', replay = 'default_replay', model = 'non_perfect', plot3d = False, plotbars = False, save = False, plot = True):
    filenames = get_all_score_filenames_in_folder(DIR)

    bss = []
    ds = []
    scores = []
    stepss = []

    depths_labels = [0,1,4,8,16,32,48]

    scores_dict = {32: np.zeros([len(depths_labels),2]) ,48: np.zeros([len(depths_labels),2]), 64: np.zeros([len(depths_labels),2]), 80: np.zeros([len(depths_labels),2])}

    for fn in filenames:
        path = fn.split('/')
        if not (path[2] == replay and path[0] == env and path[3] == num_timesteps and (path[4] == 'default' or path[4] == model)):
            continue
        if path[-3] == 'default':
            bs = int(path[-2][2:])
            d = 0
            steps = bs
        else:
            bs = int(path[-3][2:])
            d = int(path[-2][1:])
            steps = 32 + bs * d
            if steps == 47:
                steps = 48
            if steps == 65: 
                steps = 64

        if steps == 96:
            continue

        mean_rewards_dict = pickle.load(open(os.path.join(DIR,fn), 'rb'))
        
        progress = []
        for iteration in mean_rewards_dict:
            progress.append(get_progress(iteration))

        average_progresses = np.mean(progress, axis=0)
        std_progresses = np.std(progress, axis=0, ddof=1)

        step_idx = int(int(num_timesteps) / 1000 - 1)

        bss.append(bs)
        ds.append(d)
        scores.append(average_progresses[step_idx])
        stepss.append(steps)

        try:

            if env == 'MountainCar':
                scores_dict[steps][depths_labels.index(d)] = (average_progresses[step_idx]+200, std_progresses[step_idx])
            elif env == 'CartPole':
                scores_dict[steps][depths_labels.index(d)] = (average_progresses[step_idx], std_progresses[step_idx])
        except:
            continue

    if plot3d:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.set_xlabel('batchsize')
        ax.set_ylabel('depth')
        ax.set_zlabel('score')

        mapp = {32:'orange', 48:'red', 64:'green',80:'blue'}
        

        ax.scatter3D(bss, ds, scores, c=[mapp[s] for s in stepss])
        plt.show()
    
    elif plotbars:
        x = np.arange(len(depths_labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()

        widths0 = x - 1.5*width
        widths1 = np.append([-0.5*width], (x - width)[1:])
        widths2 = np.append([0.5*width], x[1:])
        widths3 = np.append([1.5*width], (x + width)[1:-1])
        widths3 = np.append(widths3, [6])


        if env == 'MountainCar':
            rects0 = ax.bar(widths0, scores_dict[32][:,0], width, yerr=scores_dict[32][:,1], bottom=-200, label='DQN32')
            rects1 = ax.bar(widths1, scores_dict[48][:,0], width, yerr=scores_dict[48][:,1], bottom=-200, label='DQN48\nAER16')
            rects2 = ax.bar(widths2, scores_dict[64][:,0], width, yerr=scores_dict[64][:,1], bottom=-200, label='DQN64\nAER32')
            rects3 = ax.bar(widths3, scores_dict[80][:,0], width, yerr=scores_dict[80][:,1], bottom=-200, label='DQN80\nAER48')

        elif env == 'CartPole':
            rects0 = ax.bar(widths0, scores_dict[32][:,0], width, yerr=scores_dict[32][:,1], label='DQN32')
            rects1 = ax.bar(widths1, scores_dict[48][:,0], width, yerr=scores_dict[48][:,1], label='DQN48\nAER16')
            rects2 = ax.bar(widths2, scores_dict[64][:,0], width, yerr=scores_dict[64][:,1], label='DQN64\nAER32')
            rects3 = ax.bar(widths3, scores_dict[80][:,0], width, yerr=scores_dict[80][:,1], label='DQN80\nAER48')

        ax.legend(bbox_to_anchor=(0.5,1), loc='upper center', ncol=4, frameon=False)
        ax.set_ylabel('Scores')
        ax.set_xlabel('Rollout length')
        
        # title generation
        model_dict = {'RM': 'reward', 'EM': 'transition', 'TM': 'termination'}
        title = env
        if model != 'non_perfect':
            title += '\nwith perfect '
            tmp = model.split('_')
            for m in tmp:
                if m == 'perfect':
                    continue
                title += model_dict[m]
                title += ', '
            title = title[:-2] + ' model'
        else:
            title += '\nwith non-perfect models'
        ax.set_title(title)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['DQN'] + depths_labels[1:])
        

        if env == 'MountainCar':
            ax.plot([-1, len(depths_labels)], [scores_dict[32][0][0]-200,scores_dict[32][0][0]-200], "b--")
            ax.set_ylim(top=-75)
        elif env == 'CartPole':
            ax.set_ylim(top=240)
            ax.plot([-1, len(depths_labels)], [scores_dict[32][0][0],scores_dict[32][0][0]], "b--")
        
        

        fig.tight_layout()
        
        if save:
            plt.savefig('./../Thesis/Figures/Scores/{}_{}.png'.format(env, model))

        if plot:
            plt.show()

# Use this to plot scores by rollout length and training ratio
def plot_all_scores(env, save, plot):
    plot_scores_by_rollout(
        env=env,
        num_timesteps='300000',
        replay='default_replay',
        model='non_perfect',
        plotbars=True,
        save=save,
        plot=plot)

    for m in ['EM','TM','RM','EM_TM_RM']: #'EM_TM','EM_RM','TM_RM'
        plot_scores_by_rollout(
        env=env,
        num_timesteps='300000',
        replay='default_replay',
        model='perfect_{}'.format(m),
        plotbars=True,
        save=save,
        plot=plot)


# Use this to plot learning curves
def get_all_progress_plots(env = 'MountainCar', num_timesteps = '300000', replay = 'default_replay', save = False, plot = True, models = ['non_perfect', 'default', 'perfect_EM', 'perfect_RM', 'perfect_TM', 'perfect_EM_TM_RM']):
    filenames = get_all_score_filenames_in_folder(DIR)
    
    titles = ['Average_mean_rewards', 'Variance_of_mean_rewards', 'Standard_deviation_of_mean_rewards', 'Average_progress', 'Variance_of_progress', 'Standard_deviation_of_progress']
    model_lables = {'non_perfect':'non-perfect', 
                    'default': 'DQN', 
                    'perfect_EM': 'perfect transition', 
                    'perfect_RM': 'perfect reward', 
                    'perfect_TM': 'perfect termination', 
                    'perfect_EM_TM_RM': 'all perfect'}


    scores_dict = dict()

    for filename in filenames:
        params = filename.split('/')

        if not (params[0] == env and params[2] == replay and params[3] == num_timesteps 
                and params[4] in models):
            continue

        model = params[4]
        step_idx = int(int(num_timesteps) / 1000)

        mean_rewards_dict = pickle.load(open(os.path.join(DIR,filename), 'rb'))

        average_progresses = dict()
        var_progresses = dict()
        std_progresses = dict()
        average_mean_rewards = dict()
        var_mean_rewards = dict()
        std_mean_rewards = dict()

        
        progress = []
        for iteration in mean_rewards_dict:
            progress.append(get_progress(iteration[1:step_idx]))
        
        average_mean_rewards = np.mean(mean_rewards_dict, axis=0)
        var_mean_rewards = np.var(mean_rewards_dict, axis=0, ddof=1)
        std_mean_rewards = np.std(mean_rewards_dict, axis=0, ddof=1)

        average_progresses = np.mean(progress, axis=0)
        var_progresses = np.var(progress, axis=0, ddof=1)
        std_progresses = np.std(progress, axis=0, ddof=1)

        # records = [average_mean_rewards,var_mean_rewards,std_mean_rewards,average_progresses,var_progresses,std_progresses]
        records = [average_progresses]
        
        if model not in scores_dict.keys():
            scores_dict[model] = []

        scores_dict[model].append(records[0])
        
    legend_label_order = ['default', 'non_perfect', 'perfect_RM', 'perfect_TM', 'perfect_EM', 'perfect_EM_TM_RM']
    legend_handles = dict()

    for key in scores_dict.keys():
        scores_dict[key] = np.mean(np.array(scores_dict[key]), axis=0)
        if key == 'default':
            legend_handles[key] = plt.plot(scores_dict[key], '--', label=model_lables[key])[0]
        else:
            legend_handles[key] = plt.plot(scores_dict[key], label=model_lables[key])[0]
        
    # plt.legend(scores_dict.keys())
    plt.legend(handles = [legend_handles[k] for k in legend_label_order], frameon=False)
    plt.title(env)
    plt.ylabel('Score')
    plt.xlabel('Time steps in $10^3$ steps')
    
    if save:
        plt.savefig('./../Thesis/Figures/Scores/{}_learning_curve.png'.format(env))


    if plot:
        plt.show()



# plot_all_scores('CartPole',save = True, plot = False)

# get_all_progress_plots(env='MountainCar', save=True)









# def get_plots(d, bs, steps):
#     episodes = []
#     rewards = []
#     with open(DIR+'{}/bs{}/d{}/'.format(steps, bs, d)+'progress.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 0:
#                 line_count += 1
#             else:
#                 episodes.append(int(row[1]))
#                 rewards.append(float(row[2]))
#                 line_count += 1
#     return episodes, rewards
