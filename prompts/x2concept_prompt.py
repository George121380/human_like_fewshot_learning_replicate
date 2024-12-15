def x2concept_prompt(x_list):
    list_str = ', '.join([str(x) for x in x_list])
    return f"""
Now you have a list of numbers: {list_str}

## Example:
# When you need to generate 3 concepts based on the list of numbers: 2, 4, 6, 8, 10
# output concepts: even numbers, numbers under or equal to 10, numbers larger than 1

## Output Fromat:
Directly output the concepts without any explaination and annotation.
"""