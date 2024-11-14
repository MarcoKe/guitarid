import torch.nn as nn
import torchvision.models as models


class BaseModelBuilder:
    def __init__(self, model_name, num_classes, freeze_weights):
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_weights = freeze_weights

    def get(self):
        # retrieve weights and instantiate
        weights = models.get_model_weights(self.model_name).DEFAULT
        model_builder = models.get_model_builder(self.model_name)
        model = model_builder(weights=weights)

        # change head
        model = self.adjust_head(model)

        # optionally freeze weights
        model = self.freeze_model_weights(model)

        return model

    # change last layer to accommodate num classes
    # may have to be overwritten for certain models if model.classifier does not exist
    def adjust_head(self, model):
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features=model.classifier[-1].in_features, out_features=self.num_classes)
        )

        return model

    # freeze all weights except head
    # may have to be overwritten for certain models if model.classifier does not exist
    def freeze_model_weights(self, model):
        if self.freeze_weights:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

        return model


class RegNetBuilder(BaseModelBuilder):
    def adjust_head(self, model):
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def freeze_model_weights(self, model):
        if self.freeze_weights:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model


# Automatically detect and map model families
def build_model_family_mapping():
    family_mapping = {}
    for family_name in dir(models):
        family = getattr(models, family_name)
        if hasattr(family, '__all__'):
            for model_name in family.__all__:
                family_mapping[model_name.lower()] = family_name.lower()
    return family_mapping


class ModelFactory:
    family_builders = {
        "regnet": RegNetBuilder,
        # Add other family-specific builders here
    }

    model_family_mapping = build_model_family_mapping()

    @staticmethod
    def create_model(model_name, num_classes, freeze_weights):
        family = ModelFactory.model_family_mapping.get(model_name.lower())
        if not family:
            raise ValueError(f"Model {model_name} is unsupported or unmapped.")

        builder_class = ModelFactory.family_builders.get(family, BaseModelBuilder)
        builder = builder_class(model_name, num_classes, freeze_weights)
        return builder.get()


if __name__ == '__main__':
    model = ModelFactory.create_model("regnet_x_8gf", 666, True)
    print(model)

