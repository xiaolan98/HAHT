import json
def load_data(path, data):
	with open(path) as json_file:
		for line in json_file.readlines():
			data.append(json.loads(line))
	return data

train_path = ["session1/train_convs.txt", "session2/train_convs.txt", "session3/train_convs.txt", "session4/train_convs.txt"]

valid_path = ["session1/valid_convs.txt", "session2/valid_convs.txt", "session3/valid_convs.txt", "session4/valid_convs.txt", "session5/valid_convs.txt"]

test_path = ["session1/test_convs.txt", "session2/test_convs.txt", "session3/test_convs.txt", "session4/test_convs.txt", "session5/test_convs.txt"]

instances = []

for _p in train_path:
	instances = load_data(_p, instances)

for _p in valid_path:
	instances = load_data(_p, instances)

for _p in test_path:
	instances = load_data(_p, instances)

print(len(instances))

total_turn = 0.0
total_conversations = 0.0

total_word = 0.0

for instance in instances:
	history_conv = instance["history_conv"]
	current_conv = instance["current_conv"]
	total_conversations += 1 + len(history_conv)
	total_turn += len(current_conv)
	for j in current_conv:
		total_word += len(j.split(" "))
	for h in history_conv:
		total_turn += len(h)
		for i in h:
			total_word += len(i.split(" "))

print(total_turn, total_conversations, total_turn/total_conversations)

print(total_word, total_turn, total_word/total_turn)