### Model select

- Codegen:
  - **Mono**: Multi + python 预训练。从给定的自然语言和编程语言文本中提取特征，并计算它们的可能性。但是，该模型旨在并且最擅长程序综合，即在给定英文提示的情况下生成可执行代码，其中提示应采用注释字符串的形式。
  - Multi: CodeGen-NL + BigQuery (Github 上各种语言的代码)
  - NL: the Pile
- all-MiniLM-L6-v2: sentence-transformer model

### Structure

- `x2concept.py`: 调用 gpt-4 / codex 的 api，为一个x_list提出数个concept假设
- `prior_model.py`: 对 all-MiniLM-L6-v2 + MLP 的封装，对codegen的封装
- `dataset.py`: Dataset类，data format: `List[tuple[List[int], int, float]]`，即$List[(X_{1:k}, X_{test}, r)]$
- `concept2Python.py`: Concept2Python类，将concept转为python程序，注意程序中判断输入x是否输入concept的函数名应为`test_function()`