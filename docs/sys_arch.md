# System Architecture

## Model Compiler Options

- Macro builder will add auxtowers in eval
- DagEdge will apply droppath in eval
- BatchNorms will be affine in eval

## Search

### Algorithm

For Darts and Random search:

```
input: conf_macro, micro_builder
output: final_desc

macro_desc = build_macro(conf_macro)
model_desc = build_desc(macro_desc, micro_builder)
model = build_model(model_desc)
train(model)
final_desc = finalize(model)
```

For PetriDish, we need to add n iteration

```
input: conf_macro, micro_builder, n_search_iter
output: final_desc

macro_desc = build_macro(conf_macro)
for i = 1 to n_search_iter:
    if pre_train_epochs > 0:
        if all nodes non-empty:
            model = build_model(model_desc, restore_state=True)
            train(mode, pre_train_epochsl)
            macro_desc = finalize(model. include_state=True)
        elif all nodes empty:
            pass because no point in training empty model
        else
            raise exception

    # we have P cells, Q nodes each with 0 edges on i=1 at this point
    # for i > 1, we have P cells, i-1 nodes at this point
    # Petridish micro builder removes 0 edges nodes after i
    # if number of nodes < i, Petridish macro adds nodes
    # assert 0 edges for all nodes for i-1
    # Petridish micro builder adds Petridish op at i
    model_desc = build_desc(macro_desc, micro_builder(i))
    # we have P cells, i node(s) each
    model = build_model(model_desc, restore_state=True)
    arch_train(model)
    macro_desc = final_desc = finalize(model. include_state=True)
    # make sure FinalPetridishOp can+will run in search mode
    # we end with i nodes in each cell for Petridish at this point
```

### Checkpointing search

Loop1: search iterations
    Loop2: pre-training
    Loop3: arch-training

Each loop has state and current index.

Cases:
    termination before Loop1
    termination before Loop2
    termination during Loop2
    termination after Loop2
    termination before Loop3
    termination during Loop3
    termination after Loop3
    termination after Loop1

Idea:
    Each node maintains its unique key in checkpoint
    Each node updates+saves checkpoint *just after* its iteration
        Checkpoint can be saved any time
    When node gets checkpoint, if it finds own key
        it restores state, iteration and continues that iteration

## Logging

We want logs to be machine readable. To that end we can think of log as dictionary. One can insert new key, value pair in this dictionary but we should allow to overwrite existing values unless value themselves are container type in which case, the log value is appended in that container. Entire log is container itself of type dictionary. ANothe container is array.

log is class derived from ordered dict. Insert values as usual. key can be option in which case internal counter may be used. It has one additional method child(key) which returns log object inserted at the key.


```
logger.add(path, val, severity=info)

path is string or tuple. If tuple then it should consist of ordered dictionary keys.

logger.add('cuda_devices', 5)
logger.add({'cuda_devices': 5, 'cuda_ver':4})
logger.add(('epochs', 5), {acc=0.9, time=4.5})
logger.add(('epochs', 5), {acc=0.9, time=4.5})

logger.begin_sec('epochs')
    logger.begin.sec(epoch_i)

        logger.add(key1, val1)
        logger.add({...})


    logger.end_Sec()
longer.end_sec()

```


## Cells and Nodes
Darts Model
    ConvBN
    Cell
        ReLUSepConv/2BN if reduction else ReLUSepConvBN
        sum for each node
        concate channels for all nodes
    AdaptiveAvgPool
    Linear
