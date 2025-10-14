import tensorflow as tf
import keras.backend as K

class VirtualTwin(tf.keras.Model):
    mean_std_scores_fields = {
        "flow_traffic",
        "link_capacity",
        "flow_propag_delay",
        "flow_length",
        "flow_loss_packet"
    }

    mean_std_scores = None

    name = "virtual_twin"

    def __init__(self, override_mean_std_scores=None, name=None):
        super(VirtualTwin, self).__init__()

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32

        if override_mean_std_scores is not None:
            self.set_mean_std_scores(override_mean_std_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        self.attention = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
            tf.keras.layers.Dense(
                self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)
            ),
            ]
        )

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate",
            ),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate",
        )

        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=4),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    )
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    )
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus)
            ],
            name="PathReadout",
        )

    def set_mean_std_scores(self, override_mean_std_scores):
        assert (
            type(override_mean_std_scores) == dict
            and all(kk in override_mean_std_scores for kk in self.mean_std_scores_fields)
            and all(len(val) == 2 for val in override_mean_std_scores.values())
        ), "overriden mean-std dict is not valid!"
        self.mean_std_scores = override_mean_std_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert self.mean_std_scores is not None, "the model cannot be called before setting the mean-std scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_length = inputs["flow_length"]
        flow_loss_packet = inputs["flow_loss_packet"]
        flow_propag_delay = inputs["flow_propag_delay"]
        link_capacity = inputs["link_capacity"]
        link_to_flow = inputs["link_to_flow"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_to_link = inputs["flow_to_link"]

        path_gather_traffic = tf.gather(flow_traffic, flow_to_link[:, :, 0])
        load = tf.math.reduce_mean(path_gather_traffic, axis=1) / link_capacity

        # Initialize the initial hidden state for paths
        path_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.mean_std_scores["flow_traffic"][0])
                    * self.mean_std_scores["flow_traffic"][1],
                    (flow_length - self.mean_std_scores["flow_length"][0])
                    * self.mean_std_scores["flow_length"][1],
                    (flow_loss_packet - self.mean_std_scores["flow_loss_packet"][0])
                    * self.mean_std_scores["flow_loss_packet"][1],
                    (flow_propag_delay - self.mean_std_scores["flow_propag_delay"][0])
                    * self.mean_std_scores["flow_propag_delay"][1],
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.mean_std_scores["link_capacity"][0])
                    * self.mean_std_scores["link_capacity"][1],
                    load
                ],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather = tf.gather(link_state, link_to_flow, name="LinkToPath")
            previous_path_state = path_state

            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )

            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, flow_to_link, name="FlowToLink"
            )

            attention_score = self.attention(path_gather)
            normalized_score = K.softmax(attention_score)
            weighted_score = normalized_score * path_gather
            path_gather_score = tf.reduce_sum(weighted_score, axis=1)

            # path_sum = tf.reduce_sum(path_gather, axis=1)

            link_state, _ = self.link_update(path_gather_score, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])

        capacity_gather = tf.gather(link_capacity, link_to_flow)

        delay = occupancy / capacity_gather
        delay = tf.math.reduce_sum(delay, axis=1)
        
        return delay + flow_propag_delay
