def write_strings_to_file(string1, string2, filename):
    try:
        with open(filename, 'w') as file:
            file.write(f'String 1: {string1}\n')
            file.write(f'String 2: {string2}\n')
        print(f'Successfully wrote strings to {filename}')
    except Exception as e:
        print(f'An error occurred: {str(e)}')

# Example usage
string1 = "Hello, this is string 1."
string2 = "And this is string 2."
filename = "output.py"

write_strings_to_file(string1, string2, filename)
