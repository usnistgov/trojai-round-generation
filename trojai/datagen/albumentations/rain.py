import albumentations.augmentations.transforms as albu


class RandomRain(albu.RandomRain):
    """Adds deterministic rain effects.
    """
    random_state_obj = None

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        slant = int(self.random_state_obj.uniform(self.slant_lower, self.slant_upper))

        height, width = img.shape[:2]
        area = height * width

        if self.rain_type == "drizzle":
            num_drops = area // 770
            drop_length = 10
        elif self.rain_type == "heavy":
            num_drops = width * height // 600
            drop_length = 30
        elif self.rain_type == "torrential":
            num_drops = area // 500
            drop_length = 60
        else:
            drop_length = self.drop_length
            num_drops = area // 600

        rain_drops = []

        for _i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = self.random_state_obj.randint(slant, width)
            else:
                x = self.random_state_obj.randint(0, width - slant)

            y = self.random_state_obj.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "rain_drops": rain_drops}
