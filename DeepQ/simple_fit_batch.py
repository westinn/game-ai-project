def fit_batch(model, gamma=0.99, start_states, actions, rewards, next_states, is_terminal)
    """Do one deep Q learning iteration.
    
    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal
    
    """
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    next_Q_values[is_terminal] = 0
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )