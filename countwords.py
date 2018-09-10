path = "tosumwords.txt"

total = 0

for line in open(path, 'r'):
    line_ = line.split()
    if line_[0] == "Added":
        total += int(line_[1])

print(total)
