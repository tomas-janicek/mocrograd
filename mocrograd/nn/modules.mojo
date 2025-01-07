from collections import Dict

from mocrograd import tensor

from . import init


alias ParametersDict = Dict[String, List[tensor.Tensor]]


trait Module:
    fn parameters(self) raises -> ParametersDict:
        ...

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        ...

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        ...

    fn __moveinit__(out self, owned existing: Self):
        ...


struct Linear(CollectionElement):
    var weights: tensor.Tensor
    var biases: tensor.Tensor

    fn __init__(
        out self,
        in_features: UInt,
        out_features: UInt,
        initialization: StringLiteral = "normal",
    ) raises:
        if initialization == "normal":
            self.weights = init.create_normal_weights(
                rows=out_features, cols=in_features
            )
            self.biases = init.create_normal_weights(rows=out_features, cols=1)
        elif initialization == "kaiming":
            self.weights = init.create_kaiming_normal_weighta(
                rows=out_features, cols=in_features
            )
            self.biases = init.create_kaiming_normal_weighta(
                rows=out_features, cols=1
            )
        else:
            raise "UnknownInitializationError"

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.weights @ input + self.biases

    fn parameters(self) -> List[tensor.Tensor]:
        return List(self.weights, self.biases)

    fn __moveinit__(out self, owned existing: Self):
        self.weights = existing.weights^
        self.biases = existing.biases^

    fn __copyinit__(out self, existing: Self):
        self.weights = existing.weights
        self.biases = existing.biases
