import albumentations.augmentations.transforms as albu


class RandomFog(albu.RandomFog):
    """Simulates fog in a deterministic way.
    """
    random_state_obj = None

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        fog_coef = self.random_state_obj.uniform(self.fog_coef_lower, self.fog_coef_upper)

        height, width = imshape = img.shape[:2]

        hw = max(1, int(width // 3 * fog_coef))

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > -hw:
            for _i in range(hw // 10 * index):
                x = self.random_state_obj.randint(midx, width - midx - hw)
                y = self.random_state_obj.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {"haze_list": haze_list, "fog_coef": fog_coef}
