import numpy as np


class Factor:
    """Class to store DEX (Design of Experiment) factor and the levels it was drawn from.
    """

    def __init__(self, levels: list[any], jitter: float = None, rso: np.random.RandomState = None, requested_level: any = None):
        """
        Initialize a Factor class instance with the provided levels. This includes sampling from those levels to pick an instance, apply any jitter, constructing the final dex factor value.

        Args:
            levels: The levels to select from for this factor.
            jitter: Any jitter to apply to the selected level before saving it into the value. I.e. value = 5.0 +- 1.2, results in a value between 3.8 and 6.2
            rso: The ransom state object used to do the sampling.
            requested_level: The requested level for this factor. This must a value within the provided levels list.
        """
        self.levels = list(set(levels))  # ensure uniqueness
        self.jitter = jitter
        self.level = None  # stores the value before any jitter
        self.value = None  # value = level +- jitter

        if rso is None:
            # create a rso object if none was provided
            rso = np.random.RandomState()

        if requested_level is not None:
            # if there is a requested value, use that in place of sampling from the levels.
            if requested_level not in self.levels:
                # confirm requested value is a valid level
                raise RuntimeError("Invalid requested_level='{}', missing from the available levels='{}'".format(requested_level, self.levels))
            # set the specific level to the index in levels
            self.level = requested_level
        else:
            # sample a level from the list of levels
            self.level = self.levels[rso.randint(len(self.levels))]

        # populate the value from the selected level
        self.value = self.level

        # apply any jitter specified
        if self.jitter is not None and self.jitter > 0:
            if isinstance(self.jitter, int):
                jitter_value = rso.randint(-self.jitter, self.jitter)
            else:
                jitter_value = rso.uniform(-self.jitter, self.jitter)
            self.value += jitter_value

