import json

json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/egoexo4d_relations/annotations/relations_val.json"
save_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/filter_takes_byname.json"

with open(json_path, "r") as fp:
    datas = json.load(fp)
datas = datas['annotations']

split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
with open(split_path, "r") as fp:
    data_split = json.load(fp)
takes = data_split["val"]


basketball = []
bike = []
cooking = []
health = []
music = []
soccer = []


for take in takes:
    anno = datas[take]
    take_name = anno["take_name"]
    take_name = take_name.lower()
    if "basketball" in take_name:
        basketball.append(take)
    elif "bike" in take_name:
        bike.append(take)
    elif "cooking" in take_name:
        cooking.append(take)
    elif "covid" in take_name:
        health.append(take)
    elif "piano" in take_name or "violin" in take_name or "guitar" in take_name:
        music.append(take)
    elif "soccer" in take_name:
        soccer.append(take)
    else:
        print("invalid take!")

takes_total = {
    "basketball":basketball,
    "bike":bike,
    "cooking":cooking,
    "health":health,
    "music":music,
    "soccer":soccer
}


with open(save_path, "w") as fp:
    json.dump(takes_total,fp)