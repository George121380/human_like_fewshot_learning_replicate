def x2concept_prompt(x_list, C_num_return):
    list_str = ', '.join([str(x) for x in x_list])
    system_prompt = f"Given a set of numbers, output {C_num_return} associated with these numbers. Try to generate diverse concepts without overlapping. Divide the concepts with comma."
    user_prompt = f"""
Now you have a list of numbers: {list_str}. Please generate {C_num_return} concepts based on the list of numbers.

## Example:
# When you need to generate 3 concepts based on the list of numbers: 2, 4, 6, 8, 10
# output concepts: even numbers, numbers under or equal to 10, numbers larger than 1

## Example:
# When you need to generate 2 concepts based on the list of numbers: 16,
# output concepts: even numbers, perfect square

## Example:
# When you need to generate 1 concepts based on the list of numbers: 8,
# output concepts: power of 2

## Example:
# When you need to generate 5 concepts based on the list of numbers: 23, 16, 19, 20
# output concepts: numbers between 15 and 25, perfect square, odd numbers, numbers under or equal to 23, numbers larger than 15

## Example:
# When you need to generate 4 concepts based on the list of numbers: 30, 10, 60, 20
# output concepts: numbers less than 100, perfect square, even numbers, divisible by 5

## Example:
# When you need to generate 3 concepts based on the list of numbers: 2, 4, 6, 8, 10
# output concepts: even numbers, numbers under or equal to 10, numbers larger than 1, multiples of 2, numbers divisible by 2, consecutive even numbers, positive integers, numbers in ascending order, numbers less than 12,  numbers ending with 0 or 2, numbers with sum of digits equal to 2, numbers with product of digits equal to 0,  numbers with no repeating digits, numbers with increasing digits, numbers with even digit sum, numbers with digit sum less than 10, numbers with digit sum greater than 1, numbers with digit sum equal to 2

## Output Fromat:
Directly output the concepts without any explaination and annotation. Divide the concepts with comma.
"""

    return system_prompt, user_prompt