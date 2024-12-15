import random

# Open file in write mode
with open('text.txt', 'w') as f:
    # Generate 100 lines
    for _ in range(100):
        # First number: random integer from 1 to 10
        n1 = random.randint(1, 10)
        # Second number: random integer from 200 to 400
        n2 = random.randint(200, 400)
        # Third number: random float between 1e-5 and 1e-4
        n3 = random.uniform(1e-5, 1e-4)
        # Write line to file in scientific notation
        f.write(f"{n1},{n2},{n3:.2e}\n")
