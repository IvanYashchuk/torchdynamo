from torchdynamo import config

from .analysis import has_mutation

def is_aot_autograd_safe_to_run(gm, example_inputs):
    """
    There are some known issues with Aot Autograd. This is a workaround to catch
    such cases, and fallback to eager. We should fix these quickly.

    Issues
    1) LSTM - https://github.com/pytorch/torchdynamo/issues/1147
    2) LSTM - https://github.com/pytorch/functorch/issues/586
    3) New op - https://github.com/pytorch/torchdynamo/issues/1448
    4) Input mutation - https://github.com/pytorch/torchdynamo/issues/1301
    """

    def raise_or_warn(reason):
        msg = f"Unable to use Aot Autograd because of presence of {reason}"
        if config.raise_on_unsafe_aot_autograd:
            raise NotImplementedError(msg)
        else:
            log.warning(msg)
        return False

    import functorch.compile

    # 1) LSTM module (tts_angular) - https://github.com/pytorch/functorch/issues/586
    for submod in gm.modules():
        if submod.__class__.__name__ == "LSTM":
            return raise_or_warn("LSTM")

    # 2) new does not work with fake tensor and aot autograd
    for node in gm.graph.nodes:
        if node.op == "call_method" and node.target == "new":
            return raise_or_warn("new operator")

    # 2) Mutation in the graph
    mutated = False
    try:
        if functorch.compile.config.use_functionalize:
            # There are two problematic classes we still exclude for now with
            # functionalization:
            #   - data mutation of inputs (fixed when we stop recording the
            #   copy_ directly into the graph)
            #   - metadata mutation of inputs (fixed if we do an extra partition
            #   to avoid AotAutograd on the mutated inputs, or if we some how
            #   get custom autograd function to reflect metadata changes to the
            #   original tensor)
            mutated = has_mutation(gm, example_inputs, inputs_only=True)
        else:
            mutated = has_mutation(gm, example_inputs)
    except NotImplementedError as e:
        if "SparseTensorImpl" not in str(e):
            # TODO - TorchDynamo mutation analysis cannot handle sparse tensors.
            # So, there is a chance that we could call Aot Autograd when it is
            # unsafe.
            # The exception is fairly guarded with string check, so any other
            # mutation analysis bugs will raise exceptions and will be caught.
            raise e
        pass

    if mutated:
        return raise_or_warn("mutation")

    return True
