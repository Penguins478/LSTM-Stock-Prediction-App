file = open("data.txt", "r")

lines = file.readlines()

total = 0
count = 0
for line in lines:
    str_arr = line.split()
    val = float(str_arr[1])
    count += 1
    total += val

average = total / count

print("Overall Accuracy: " + str(average))