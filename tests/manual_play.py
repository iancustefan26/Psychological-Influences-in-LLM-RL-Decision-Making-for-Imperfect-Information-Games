import rlcard
import torch
from rlcard.agents import LimitholdemHumanAgent
from rlcard.agents.dqn_agent import DQNAgent

# 1. Setup the environment (2 players)
# Note: 'record_action' is useful for human play to see the game history
env = rlcard.make('limit-holdem', config={'num_players': 2, 'record_action': True})

# 2. Load the trained DQN Agent
print("Loading DQN Agent...")
checkpoint_dict = torch.load('checkpoints/dqn_2players.pt', map_location='cpu', weights_only=False)

dqn_agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[512, 256, 128], 
    device='cpu'
)
dqn_agent.from_checkpoint(checkpoint_dict)

# 3. Initialize the Human Agent
human_agent = LimitholdemHumanAgent(env.num_actions)

# 4. Bind agents (You are Player 0, DQN is Player 1)
env.set_agents([human_agent, dqn_agent])

print("\n>> Game Setup Complete. You are Player 0.")

# 5. Interactive Game Loop
while True:
    print("\n" + "="*40)
    print("STARTING NEW HAND")
    print("="*40)
    
    # is_training=False ensures the DQN doesn't take random exploratory actions
    trajectories, payoffs = env.run(is_training=False)
    
    # Extract the DQN agent's cards directly from the game engine
    # env.game.players[1].hand contains the Card objects for Player 1 (the DQN)
    # card.get_index() converts the object to a readable string like 'HQ' (Heart Queen)
    dqn_cards = [card.get_index() for card in env.game.players[1].hand]
    
    # It is also helpful to pull the final community board for context
    public_cards = [card.get_index() for card in env.game.public_cards]
    
    print("\n--- Hand Over ---")
    print(f"Community Cards:        {public_cards}")
    print(f"DQN's Hole Cards:       {dqn_cards}")
    print("-------------------------")
    print(f"Your Payoff (Player 0): {payoffs[0]}")
    print(f"DQN Payoff (Player 1):  {payoffs[1]}")
    
    # Prompt to play another hand
    val = input("\nPress Enter to play another hand, or type 'q' to quit: ")
    if val.strip().lower() == 'q':
        break