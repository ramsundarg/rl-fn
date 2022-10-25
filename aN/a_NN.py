from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
def get_actor_net(action_spec, actor_activation_fn, actor_fc_layers, observation_spec):
    return actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        fc_layer_params=actor_fc_layers,
        dropout_layer_params=None,
        conv_layer_params=None,
        activation_fn=actor_activation_fn,
        kernel_initializer=None,
        last_kernel_initializer=None)