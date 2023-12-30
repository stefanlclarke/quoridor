class QuoridoorAgent:
    def __init__(self, input_type='board', output_type='one_hot'):
        """
        Generic agent class.

        input_type: either 'board' or 'game'
        output_type: either 'one_hot' or 'true'
        """

        self.input_type = input_type
        self.output_type = output_type

    def move(self, input):
        """
        Should make the move for the agent.

        Returns a numpy array in the format dictated by the agent
        output type.
        """
        raise NotImplementedError()
