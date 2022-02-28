import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input

from l2rpn_baselines.utils import TrainingParam
from agents.utils.SingularBase import SingularBaseQ


class DQNPER_NN(SingularBaseQ):
    """Constructs the desired deep q learning network"""
    def __init__(self, nn_params, training_param=None):

        if training_param is None:
            training_param = TrainingParam()

        SingularBaseQ.__init__(self, nn_params, training_param)

        self.construct_q_network()

    def construct_q_network(self):
        """
        It uses the architecture defined in the `nn_archi` attributes.

        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,), name="observation")

        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        output_layer = Dense(self._action_size, name="output")(lay)

        self._model = Model(inputs=[input_layer], outputs=[output_layer])
        self._schedule_model, self._optimizer_model = self.make_optimizer()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)
