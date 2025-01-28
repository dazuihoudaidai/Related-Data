def find_largest_values_with_position(file_path, num_values=13000):

    with open(file_path, 'r') as file:
        lines = file.readlines()

        numbers_per_line = [float(value) for line in lines for value in line.strip().split()]


    largest_values_with_position = sorted(enumerate(numbers_per_line, start=1), key=lambda x: x[1], reverse=True)[:num_values]

    return largest_values_with_position


file_path = r""
largest_values_with_position = find_largest_values_with_position(file_path)


for i, (position, value) in enumerate(largest_values_with_position, start=1):
    print(f"The {i}th largest value is: {value}, at position {position} in the file")
