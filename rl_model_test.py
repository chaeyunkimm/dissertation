import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import src.metropolis_hastings as mh

from src.time_varying_vine_copula import tv_vinecop, obs_transform_pendulum, action_transform_pendulum

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5")  
    # Reset environment to start a new episode
    observation, info = env.reset(seed=123) 
    # observation: what the agent can "see" - cart position, velocity
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")

    n = 200 #truncation limit
    obs_d = env.observation_space.shape[0]
    act_d = env.action_space.shape[0]
    print(obs_d, act_d)

    episode_over = False
    total_reward = 0

    print(obs_d, act_d)
    observations = np.zeros((n+1, obs_d))
    actions = np.zeros((n+1, act_d))
    set_actions = np.genfromtxt('./actions.csv', delimiter=',')
    set_a_l = len(set_actions)
    actions[:set_a_l,0] = set_actions
    observations[0] = observation
    i=1
    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        if i>set_a_l:
            actions[i-1] = env.action_space.sample()  # Random action for now - real agents will be smarter!
        # Take the action and see what happens
        observations[i], reward, terminated, truncated, info = env.step(actions[i-1])

        total_reward += reward
        episode_over = terminated or truncated
        i+=1
    if i <= n+1:
        observations = observations[:i]
        actions = actions[:i]

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

    data = np.concatenate((observations, actions), axis=1)

    obs_transformer = obs_transform_pendulum(standardise=False)
    act_transformer = action_transform_pendulum(standardise=False)

    obs_transformer.fit(torch.from_numpy(observations))
    act_transformer.fit(torch.from_numpy(actions))

    trans_obs = obs_transformer.transform(torch.from_numpy(observations)).numpy()
    trans_acts = act_transformer.transform(torch.from_numpy(actions)).numpy()

    trans_data = np.concatenate((trans_obs, trans_acts), axis=1)

    print("Starting Fit")
    tv = tv_vinecop(observation_d=5)
    tv.fit(trans_data, check_vines=False)
    print("Fit finished")

    tv.step_forward(np.array([trans_obs[0]]), trans_acts[0][0])
    print(tv.last_state.shape)
    print(np.array([trans_obs[0]]))

    print("Starting Metropolis Hastings")
    predicted_next = np.zeros_like(observations)
    predicted_up_err = np.zeros_like(observations)
    predicted_down_err = np.zeros_like(observations)
    predicted_next[0] = observations[0]

    for i, act in enumerate(trans_acts[0:-1]):
        # target = target_func(tv, act[0], log=True)
        # #chain, ar = mh.metropolis_hastings(target, prop, sample, np.array(observations[i+2]), symmetric_proposal=True, chain_length=800, burn_in=500)
        # chain = mh.adaptive_mh(target, torch.tensor(predicted_next[i], dtype = torch.double), torch.eye(4, dtype=torch.double)*var, nmoves = 700, return_entire_chain=True, adapt_no=100, burn_in=400)

        pred, pred_std, ar, fig = tv.predict_next_state(action = act[0], covariance=0.0005, chain_length=600, burn_in = 300, check_chain=False)

        # fig.savefig(f"C:/Users/woodg/Documents/Vine_dissertation_chaeyun/plots_rl/chain{i}")
        # plt.show()
        # plt.close()

        predicted_next[i+1] = obs_transformer.inverse_transform(torch.from_numpy(np.expand_dims(pred, axis=0)))
        predicted_up_err[i+1] = obs_transformer.inverse_transform(torch.from_numpy(np.expand_dims(pred+pred_std, axis=0)))
        predicted_down_err[i+1] = obs_transformer.inverse_transform(torch.from_numpy(np.expand_dims(pred-pred_std, axis=0)))

        # Calculate prediction, prediction variance
        # predicted_next[i+1] = torch.mean(chain, axis = 0)
        # predicted_std[i+1] = torch.std(chain, axis = 0)

        print(f"Step: {i}, Action: {act[0]}, Acceptance rate {ar}")

        # We would draw a sample from the policy here to determine the action.

        # Move the tv_vinecop to the last prediction and the action given that state.
        tv.step_forward(np.array([pred]), trans_acts[i+1][0])

fig, ax = plt.subplots(2,4)
x = np.arange(len(predicted_next))
#Plot the 
for i in range(4):
    ax[0, i].plot(x, observations[:,i], label = "Observed")
    ax[0, i].plot(x, predicted_next[:,i], label = "Prediction")
    ax[0, i].fill_between(
        x,
        predicted_up_err[:, i],
        predicted_down_err[:, i],
        color = "orange",
        alpha=0.25,
        label="Prediction uncertainty"
    )
    ax[1, i].plot(observations[:,i], predicted_next[:,i] - observations[:,i], '.')

#print(observations[-11], observations[-10], np.mean(chain, axis=0), observations[-9])
plt.savefig("./plots_rl/roll_out_no_observations_transformed_unstandardised")
plt.close()