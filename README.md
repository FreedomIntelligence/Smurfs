## Smurfs

### Inference

运行测试
```python
sbatch Inference/inference_decompose.sh
```
处理结果会保存在data中

然后
```python
python transform.py
```
来处理原始推理数据，获取符合toolbench格式的测试数据

各个agent的prompt位于Prompts/decompose_prompt.py中

toolbench的evaluation详见 [tooleval](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval)

### Evaluation

在tooleval/model_predictions_converted中创建一个文件夹，用你的模型名字命名。把inference得到的transform过的数据放入。

tooleval/utils.py中修改test_sets列表来选择测哪几个数据集，本project只选择G2_category和G3_instruction进行测试

#### Pass Rate
在tooleval/run_pass_rate.sh中修改CANDIDATE_MODEL为你创建的文件夹的名字

然后运行
```shell
bash tooleval/run_pass_rate.sh
```

#### Win Rate
在tooleval/run_preference_rate.sh中修改CANDIDATE_MODEL为你创建的文件夹的名字

然后运行
```shell
bash tooleval/run_preference_rate.sh
```
终端上会显示Win rate和Tie rate
根据toolllama的论文，最终的Win rate由：
Win rate = Win rate + 1/2 Tie rate
计算得到

#### Success Rate
在tooleval/run_success_rate.sh中修改CANDIDATE_MODEL为你创建的文件夹的名字

然后运行
```shell
bash tooleval/run_success_rate.sh
```

评测结果：[见谷歌表格](https://docs.google.com/spreadsheets/d/1RKoWmKt2abNGVzbRJDsuYYQmkV7l2IMOLodQKaZOEJA/edit?usp=sharing)