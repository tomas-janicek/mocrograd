from collections import Dict

from mocrograd import tensor

from . import modules


struct MLP(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=2,
            out_features=16,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=16,
            out_features=16,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=16,
            out_features=1,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        return out

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigits(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=64,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=64,
            out_features=32,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=32,
            out_features=10,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^


struct MLPDigitsBigger(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear
    var l4: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=8192,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=8192,
            out_features=4096,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=4096,
            out_features=2048,
            initialization="kaiming",
        )
        self.l4 = modules.Linear(
            in_features=2048,
            out_features=10,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        out = out.relu()
        out = self.l4.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()
        parameters["l4"] = self.l4.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^
        self.l4 = existing.l4^


struct MLPDigitsLonger(modules.Module):
    var l1: modules.Linear
    var l2: modules.Linear
    var l3: modules.Linear
    var l4: modules.Linear
    var l5: modules.Linear
    var l6: modules.Linear
    var l7: modules.Linear
    var l8: modules.Linear
    var l9: modules.Linear
    var l10: modules.Linear
    var l11: modules.Linear
    var l12: modules.Linear
    var l13: modules.Linear
    var l14: modules.Linear
    var l15: modules.Linear
    var l16: modules.Linear
    var l17: modules.Linear
    var l18: modules.Linear
    var l19: modules.Linear
    var l20: modules.Linear
    var l21: modules.Linear
    var l22: modules.Linear
    var l23: modules.Linear
    var l24: modules.Linear
    var l25: modules.Linear
    var l26: modules.Linear
    var l27: modules.Linear
    var l28: modules.Linear
    var l29: modules.Linear
    var l30: modules.Linear

    fn __init__(inout self) raises -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=256,
            initialization="kaiming",
        )
        self.l2 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l3 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l4 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l5 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l6 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l7 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l8 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l9 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l10 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l11 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l12 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l13 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l14 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l15 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l16 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l17 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l18 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l19 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l20 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l21 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l22 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l23 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l24 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l25 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l26 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l27 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l28 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l29 = modules.Linear(
            in_features=256,
            out_features=256,
            initialization="kaiming",
        )
        self.l30 = modules.Linear(
            in_features=256,
            out_features=10,
            initialization="kaiming",
        )

    fn __call__(self, input: tensor.Tensor) raises -> tensor.Tensor:
        return self.forward(input)

    fn forward(self, input: tensor.Tensor) raises -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        out = out.relu()
        out = self.l4.forward(out)
        out = out.relu()
        out = self.l5.forward(out)
        out = out.relu()
        out = self.l6.forward(out)
        out = out.relu()
        out = self.l7.forward(out)
        out = out.relu()
        out = self.l8.forward(out)
        out = out.relu()
        out = self.l9.forward(out)
        out = out.relu()
        out = self.l10.forward(out)
        out = out.relu()
        out = self.l11.forward(out)
        out = out.relu()
        out = self.l12.forward(out)
        out = out.relu()
        out = self.l13.forward(out)
        out = out.relu()
        out = self.l14.forward(out)
        out = out.relu()
        out = self.l15.forward(out)
        out = out.relu()
        out = self.l16.forward(out)
        out = out.relu()
        out = self.l17.forward(out)
        out = out.relu()
        out = self.l18.forward(out)
        out = out.relu()
        out = self.l19.forward(out)
        out = out.relu()
        out = self.l20.forward(out)
        out = out.relu()
        out = self.l21.forward(out)
        out = out.relu()
        out = self.l22.forward(out)
        out = out.relu()
        out = self.l23.forward(out)
        out = out.relu()
        out = self.l24.forward(out)
        out = out.relu()
        out = self.l25.forward(out)
        out = out.relu()
        out = self.l26.forward(out)
        out = out.relu()
        out = self.l27.forward(out)
        out = out.relu()
        out = self.l28.forward(out)
        out = out.relu()
        out = self.l29.forward(out)
        out = out.relu()
        out = self.l30.forward(out)

        log_probabilities = out.log_softmax()
        return log_probabilities

    fn parameters(self) raises -> modules.ParametersDict:
        var parameters = modules.ParametersDict()

        parameters["l1"] = self.l1.parameters()
        parameters["l2"] = self.l2.parameters()
        parameters["l3"] = self.l3.parameters()
        parameters["l4"] = self.l4.parameters()
        parameters["l5"] = self.l5.parameters()
        parameters["l6"] = self.l6.parameters()
        parameters["l7"] = self.l7.parameters()
        parameters["l8"] = self.l8.parameters()
        parameters["l9"] = self.l9.parameters()
        parameters["l10"] = self.l10.parameters()
        parameters["l11"] = self.l11.parameters()
        parameters["l12"] = self.l12.parameters()
        parameters["l13"] = self.l13.parameters()
        parameters["l14"] = self.l14.parameters()
        parameters["l15"] = self.l15.parameters()
        parameters["l16"] = self.l16.parameters()
        parameters["l17"] = self.l17.parameters()
        parameters["l18"] = self.l18.parameters()
        parameters["l19"] = self.l19.parameters()
        parameters["l20"] = self.l20.parameters()
        parameters["l21"] = self.l21.parameters()
        parameters["l22"] = self.l22.parameters()
        parameters["l23"] = self.l23.parameters()
        parameters["l24"] = self.l24.parameters()
        parameters["l25"] = self.l25.parameters()
        parameters["l26"] = self.l26.parameters()
        parameters["l27"] = self.l27.parameters()
        parameters["l28"] = self.l28.parameters()
        parameters["l29"] = self.l29.parameters()
        parameters["l30"] = self.l30.parameters()

        return parameters

    fn __moveinit__(inout self, owned existing: Self):
        self.l1 = existing.l1^
        self.l2 = existing.l2^
        self.l3 = existing.l3^
        self.l4 = existing.l4^
        self.l5 = existing.l5^
        self.l6 = existing.l6^
        self.l7 = existing.l7^
        self.l8 = existing.l8^
        self.l9 = existing.l9^
        self.l10 = existing.l10^
        self.l11 = existing.l11^
        self.l12 = existing.l12^
        self.l13 = existing.l13^
        self.l14 = existing.l14^
        self.l15 = existing.l15^
        self.l16 = existing.l16^
        self.l17 = existing.l17^
        self.l18 = existing.l18^
        self.l19 = existing.l19^
        self.l20 = existing.l20^
        self.l21 = existing.l21^
        self.l22 = existing.l22^
        self.l23 = existing.l23^
        self.l24 = existing.l24^
        self.l25 = existing.l25^
        self.l26 = existing.l26^
        self.l27 = existing.l27^
        self.l28 = existing.l28^
        self.l29 = existing.l29^
        self.l30 = existing.l30^
