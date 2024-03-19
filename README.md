## Multi_agents

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
